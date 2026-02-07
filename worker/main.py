"""Worker process entry point."""
import asyncio
from .daemon import RagWorker


async def main():
    """Main entry point for worker."""
    worker = RagWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

