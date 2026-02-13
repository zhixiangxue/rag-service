"""
Domain models for rag-service
"""

from .mortgage import (
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
    # Queries
    UserConditions,
    EligibilityResult,
    ProgramComparison,
)

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
]
