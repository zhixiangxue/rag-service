"""
Debug script for mortgage graph extraction and query.

This script demonstrates the complete workflow:
1. Read PDF document using MinerU
2. Extract program structure using LLM
3. Store in FalkorDB graph database
4. Query for eligibility matching

Usage:
    cd c:\\Users\\xue\\PycharmProjects\\zag-ai\\rag-service
    ..\\.venv\\Scripts\\python -m worker.debug.debug_mortgage_graph
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path (allowed in debug scripts)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "rag-service"))

from zag.readers import MinerUReader
from zag.storages.graph import FalkorDBGraphStorage

from domain.mortgage import (
    MortgageGraphStorage,
    ProgramMatcher,
    UserConditions,
    PropertyType,
    TransactionType,
    Program,
    Product,
    Rule,
    Matrix,
)
from worker.indexers.graph import (
    MortgageProgramExtractor,
    EligibilityCalculator,
)


class ProgramComparator:
    """Simple comparator using ProgramMatcher - for demo purposes."""
    def __init__(self, storage):
        self.storage = storage
        self.matcher = ProgramMatcher(storage)
    
    def compare(self, conditions, limit=5):
        matches = self.matcher.match(conditions, limit=limit)
        return SimpleComparisonResult(matches)


class SimpleComparisonResult:
    """Simple comparison result for demo."""
    def __init__(self, matches):
        self.matches = matches
        self.summary = f"Found {len(matches)} matching programs"
        self.best_option = matches[0] if matches else None


async def main():
    """Main demo flow."""
    print("=" * 60)
    print("Mortgage Program Graph - Debug Script")
    print("=" * 60)
    
    # Configuration - adjust path as needed
    pdf_path = project_root / "downloads/JMAC Lending/JMAC-DSCR-Prime.pdf"
    
    # Check file exists
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        print("Please update the path or download the PDF first.")
        return
    
    # ========================================
    # Step 1: Read PDF
    # ========================================
    print("\n[Step 1] Reading PDF document...")
    reader = MinerUReader(backend="pipeline")
    doc = reader.read(str(pdf_path))
    print(f"Document read: {len(doc.content)} characters, {len(doc.pages)} pages")
    
    # ========================================
    # Step 2: Extract Program Structure
    # ========================================
    print("\n[Step 2] Extracting program structure...")
    
    extractor = MortgageProgramExtractor(
        llm_uri="openai/gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    result = await extractor.extract(doc.content, source_file=str(pdf_path))
    
    print(f"Extraction completed: {len(result.stages_completed)} stages")
    if result.program:
        print(f"  Program: {result.program.name} by {result.program.lender}")
    print(f"  Products: {len(result.products)}")
    print(f"  Matrices: {len(result.matrices)}")
    print(f"  Cells: {result.cell_count} (across all matrices)")
    print(f"  Rules: {len(result.rules)}")
    print(f"    - State-specific: {len(result.state_overlays)}")
    
    if result.errors:
        print(f"  Errors: {result.errors}")
    
    # ========================================
    # Step 3: Store in Graph Database
    # ========================================
    print("\n[Step 3] Storing in graph database...")
    
    try:
        base_storage = FalkorDBGraphStorage(
            host="localhost",
            port=6379,
            graph_name="mortgage_demo",
        )
        base_storage.connect()
        print("Connected to FalkorDB")
        
        # Wrap with mortgage-specific storage
        storage = MortgageGraphStorage(base_storage)
        
        # Clear existing data
        base_storage.clear()
        
        # Store program data
        program_id = storage.store_program(result.to_graph_data())
        print(f"Program stored with ID: {program_id}")
        
        # ========================================
        # Step 4: Query for Eligibility
        # ========================================
        print("\n[Step 4] Querying for eligibility...")
        
        # Create user conditions
        user_conditions = UserConditions(
            fico=730,
            dscr=1.15,
            loan_amount=850000,
            state="CA",
            property_type=PropertyType.SFR,
            transaction_type=TransactionType.PURCHASE,
        )
        
        print(f"User conditions: FICO={user_conditions.fico}, DSCR={user_conditions.dscr}")
        print(f"                 Loan=${user_conditions.loan_amount:,}, State={user_conditions.state}")
        
        # Use ProgramMatcher
        matcher = ProgramMatcher(base_storage)
        matches = matcher.match(user_conditions, limit=5)
        
        print(f"\nFound {len(matches)} matching programs:")
        for i, match in enumerate(matches, 1):
            status = "ELIGIBLE" if match.eligible else "NOT ELIGIBLE"
            ltv = f"{match.max_ltv}%" if match.max_ltv else "N/A"
            print(f"  {i}. {match.program_name} - {status} (LTV: {ltv})")
        
        # Use ProgramComparator for comparison
        print("\n[Step 5] Comparing programs...")
        comparator = ProgramComparator(base_storage)
        comparison = comparator.compare(user_conditions, limit=5)
        
        print(f"\nComparison summary:")
        print(comparison.summary)
        
        if comparison.best_option:
            print(f"\nRecommended: {comparison.best_option.program_name}")
            print(f"  Max LTV: {comparison.best_option.max_ltv}%")
            print(f"  Score: {comparison.best_option.match_score:.2f}")
        
        base_storage.disconnect()
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Make sure FalkorDB is running on localhost:6379")
    
    print("\n" + "=" * 60)
    print("Debug completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
