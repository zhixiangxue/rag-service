"""
Mortgage domain models
"""

from .graph import (
    # Enums
    ProgramStatus,
    ProductType,
    TransactionType,
    BorrowerType,
    EntityType,
    PropertyType,
    OccupancyType,
    OperatorType,
    # Base
    GraphNode,
    # Core models
    Rule,
    Cell,
    Matrix,
    DocumentRequirement,
    ARMParams,
    Product,
    Program,
    # Extraction
    ExtractionStage,
    FullExtractionResult,
)
from .queries import (
    UserConditions,
    EligibilityResult,
    ProgramComparison,
)
from .storage import MortgageGraphStorage
from .matcher import ProgramMatcher, MatchResult

__all__ = [
    # Enums
    "ProgramStatus",
    "ProductType",
    "TransactionType",
    "BorrowerType",
    "EntityType",
    "PropertyType",
    "OccupancyType",
    "OperatorType",
    # Base
    "GraphNode",
    # Core models
    "Rule",
    "Cell",
    "Matrix",
    "DocumentRequirement",
    "ARMParams",
    "Product",
    "Program",
    # Extraction
    "ExtractionStage",
    "FullExtractionResult",
    # Queries
    "UserConditions",
    "EligibilityResult",
    "ProgramComparison",
    # Storage & Matching
    "MortgageGraphStorage",
    "ProgramMatcher",
    "MatchResult",
]
