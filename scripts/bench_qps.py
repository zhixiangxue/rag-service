"""
QPS benchmark for the RAG query endpoint.

Sends concurrent query requests at increasing concurrency levels and reports:
  - Requests per second (QPS)
  - Latency p50 / p95 / p99
  - Error rate

Usage:
    python playground/bench_qps.py
    python playground/bench_qps.py --concurrency 1 5 10 20 --rounds 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from client import RagClient  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────

# API_BASE   = "http://localhost:8000/" #local
# API_BASE   = "http://13.56.109.233:8000/" #dev
API_BASE   = "http://54.215.128.27:8000/" #prod
DATASET_ID = "mvUisSWfRQx2Ap836jyIy"
TIMEOUT    = 60.0

# Queries to rotate through (avoids cache-hit bias)
QUERIES = [
    "loan to value ratio maximum",
    "minimum credit score requirements",
    "debt to income ratio limit",
    "self employed borrower documentation",
    "maximum loan amount conforming",
    "prepayment penalty terms",
    "interest rate cap adjustable",
    "property appraisal requirements",
    "gift funds down payment",
    "cash out refinance guidelines",
]

# Concurrency levels to test (number of parallel in-flight requests)
DEFAULT_CONCURRENCY = [1, 10, 30]

# Requests per concurrency level
DEFAULT_ROUNDS = 10


# ── Benchmark helpers ──────────────────────────────────────────────────────────

async def single_request(client: RagClient, query: str) -> tuple[float, bool]:
    """Send one query and return (latency_seconds, success)."""
    t0 = time.perf_counter()
    try:
        results = await client.query(query, top_k=5)
        ok = results is not None
    except Exception as exc:
        print(f"    [ERR] {type(exc).__name__}: {exc}", flush=True)
        ok = False
    return time.perf_counter() - t0, ok


async def run_level(
    client: RagClient,
    concurrency: int,
    rounds: int,
) -> dict:
    """
    Run `rounds` requests at `concurrency` parallel workers.

    Each worker fires requests sequentially; all workers run concurrently.
    Total requests = rounds * concurrency.
    """
    latencies: list[float] = []
    errors: int = 0

    async def worker(queries_slice: list[str]) -> None:
        nonlocal errors
        for q in queries_slice:
            lat, ok = await single_request(client, q)
            latencies.append(lat)
            if not ok:
                errors += 1
            # Real-time progress so we can see requests are actually being sent
            print(f"    req#{len(latencies):>3}  {'ok' if ok else 'ERR'}  {lat:.2f}s", flush=True)

    # Distribute rounds across workers
    all_queries = [QUERIES[i % len(QUERIES)] for i in range(rounds * concurrency)]
    chunks = [all_queries[i::concurrency] for i in range(concurrency)]

    wall_start = time.perf_counter()
    await asyncio.gather(*[worker(chunk) for chunk in chunks])
    wall_elapsed = time.perf_counter() - wall_start

    total = len(latencies)
    qps   = total / wall_elapsed if wall_elapsed > 0 else 0

    sorted_lat = sorted(latencies)
    p50 = statistics.median(sorted_lat)
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0

    return {
        "concurrency": concurrency,
        "total":       total,
        "errors":      errors,
        "qps":         qps,
        "p50":         p50,
        "p95":         p95,
        "p99":         p99,
        "wall":        wall_elapsed,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def fmt(v: float) -> str:
    return f"{v:.3f}s"


async def main(concurrency_levels: list[int], rounds: int) -> None:
    print("=" * 72)
    print(f"  RAG Query QPS Benchmark")
    print(f"  API     : {API_BASE}")
    print(f"  Dataset : {DATASET_ID}")
    print(f"  Rounds  : {rounds} per worker per level")
    print(f"  Levels  : {concurrency_levels}")
    print("=" * 72)

    header = f"{'concurrency':>12}  {'total':>6}  {'errors':>6}  {'QPS':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}"
    print(f"\n{header}")
    print("─" * len(header))

    client = RagClient(API_BASE, DATASET_ID, timeout=TIMEOUT)
    results = []

    try:
        # Warm-up: one request to init connections
        print("  Warming up...", end=" ", flush=True)
        await single_request(client, QUERIES[0])
        print("done\n")

        for c in concurrency_levels:
            print(f"  Running concurrency={c} ...", end=" ", flush=True)
            r = await run_level(client, c, rounds)
            results.append(r)
            print(
                f"\r  {r['concurrency']:>12}  {r['total']:>6}  {r['errors']:>6}  "
                f"{r['qps']:>7.2f}  {fmt(r['p50']):>7}  {fmt(r['p95']):>7}  {fmt(r['p99']):>7}"
            )

    finally:
        await client.close()

    # Summary
    if results:
        best = max(results, key=lambda x: x["qps"])
        print(f"\n  Peak QPS = {best['qps']:.2f}  at concurrency={best['concurrency']}")
        print(f"  (p99 at peak = {fmt(best['p99'])})")

    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=DEFAULT_CONCURRENCY,
        metavar="N",
        help="concurrency levels to test (default: 1 5 10 20 40)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
        help="requests per worker per level (default: 30)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.rounds))
