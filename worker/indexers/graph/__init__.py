"""
Graph-based indexer for mortgage programs.

This module provides graph-based indexing functionality:
- Extractor: Multi-stage LLM extraction for mortgage documents
- Calculator: Precise parameter calculation
- Indexer: Full indexing pipeline

Note: MortgageGraphStorage and ProgramMatcher are now in domain/mortgage/
"""

from .extractor import MortgageProgramExtractor
from .calculator import EligibilityCalculator
from .indexer import index_graph

# Re-export from domain for backward compatibility
from domain.mortgage import MortgageGraphStorage, ProgramMatcher, MatchResult

__all__ = [
    "MortgageProgramExtractor",
    "ProgramMatcher",
    "EligibilityCalculator",
    "index_graph",
    "MortgageGraphStorage",
    "MatchResult",
]
