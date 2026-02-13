"""
Mortgage Program Graph Schemas

This module defines the data models for representing mortgage program rules
as a graph structure. The design supports complex condition logic including:
- Simple threshold conditions (FICO >= 720)
- Range conditions (DSCR in [1.0, 1.149])
- IF-THEN rules
- State-level overlays
- Multi-dimensional eligibility matrices
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class ProgramStatus(str, Enum):
    """Program status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DEPRECATED = "deprecated"


class ProductType(str, Enum):
    """Product type classification"""
    FIXED = "Fixed"
    ARM = "ARM"  # Adjustable Rate Mortgage
    SOFR_ARM = "SOFR_ARM"
    HYBRID = "Hybrid"
    JUMBO = "Jumbo"
    DSCR = "DSCR"
    NON_QM = "Non_QM"


class TransactionType(str, Enum):
    """Transaction type"""
    PURCHASE = "Purchase"
    RATE_TERM = "Rate_Term"  # Rate/Term Refinance
    CASH_OUT = "Cash_Out"  # Cash-Out Refinance
    DELAYED_FINANCING = "Delayed_Financing"
    CONSTRUCTION = "Construction"


class BorrowerType(str, Enum):
    """Borrower type classification"""
    US_CITIZEN = "US_Citizen"
    PERMANENT_RESIDENT = "Permanent_Resident"
    NON_PERMANENT_RESIDENT = "Non_Permanent_Resident"
    FOREIGN_NATIONAL = "Foreign_National"
    DACA = "DACA"
    ITIN = "ITIN"  # Individual Taxpayer Identification Number


class EntityType(str, Enum):
    """Entity type for vesting"""
    LLC = "LLC"
    PARTNERSHIP = "Partnership"
    CORPORATION = "Corporation"
    S_CORPORATION = "S_Corporation"
    INDIVIDUAL = "Individual"


class PropertyType(str, Enum):
    """Property type classification"""
    SFR = "SFR"  # Single Family Residence
    SFR_ATTACHED = "SFR_Attached"
    TWO_UNITS = "2_Units"
    THREE_UNITS = "3_Units"
    FOUR_UNITS = "4_Units"
    FIVE_EIGHT_UNITS = "5_8_Units"
    PUD = "PUD"  # Planned Unit Development
    CONDO = "Condo"
    CONDO_HOTEL = "Condo_Hotel"
    NON_WARRANTABLE_CONDO = "Non_Warrantable_Condo"
    MIXED_USE = "Mixed_Use"
    MODULAR = "Modular"
    LEASEHOLD = "Leasehold"


class OccupancyType(str, Enum):
    """Occupancy type"""
    PRIMARY = "Primary"
    SECOND_HOME = "Second_Home"
    INVESTMENT = "Investment"


class OperatorType(str, Enum):
    """Comparison operators"""
    EQ = "="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"


# =============================================================================
# Base Models
# =============================================================================


class GraphNode(BaseModel):
    """Base class for all graph nodes"""
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created_at: Optional[str] = None
    source_page: Optional[int] = None
    source_text: Optional[str] = None

    def to_cypher_props(self, exclude_none: bool = True) -> str:
        """Convert to Cypher property string with proper escaping"""
        data = self.model_dump(exclude_none=exclude_none)
        props = []
        for k, v in data.items():
            if v is None and exclude_none:
                continue
            if isinstance(v, str):
                escaped = v.replace("'", "\\'")
                props.append(f"{k}: '{escaped}'")
            elif isinstance(v, bool):
                props.append(f"{k}: {str(v).lower()}")
            elif isinstance(v, (int, float)):
                props.append(f"{k}: {v}")
            elif isinstance(v, list):
                items = ", ".join(f"'{item}'" if isinstance(item, str) else str(item) for item in v)
                props.append(f"{k}: [{items}]")
            elif isinstance(v, Enum):
                props.append(f"{k}: '{v.value}'")
            else:
                props.append(f"{k}: '{str(v)}'")
        return "{ " + ", ".join(props) + " }"


# =============================================================================
# Rule Models
# =============================================================================


class Rule(GraphNode):
    """
    Simple IFTTT-style rule for LLM-based reasoning.
    
    Designed for natural language processing by LLM rather than programmatic evaluation.
    
    Examples:
    - trigger: "Loan in California (CA)" -> action: "max_ltv capped at 70%"
    - trigger: "Loan amount exceeds $2,000,000" -> action: "Additional documents required"
    
    Relationships:
        (Program)-[:HAS_RULE]->(Rule)
        (Product)-[:HAS_RULE]->(Rule)
    """
    name: str = Field(..., description="Rule name/title")
    category: str = Field(..., description="Rule category: state, borrower, loan, property, transaction, general")
    trigger: str = Field(..., description="Natural language trigger condition (IF part)")
    action: str = Field(..., description="Natural language action/constraint (THEN part)")
    excludes: bool = Field(False, description="Whether this rule completely excludes eligibility")
    priority: int = Field(100, description="Priority for rule evaluation")
    source_text: Optional[str] = Field(None, description="Original text from document")


# =============================================================================
# Matrix Models
# =============================================================================


class Cell(GraphNode):
    """
    Single cell in an eligibility matrix.
    
    Represents a decision point in a multi-dimensional matrix.
    Example: FICO 720+ AND DSCR 1.0-1.149 AND Loan <= $1M => LTV 75%
    
    Relationships:
        (Matrix)-[:HAS_CELL]->(Cell)
    """
    # Input dimensions (ranges)
    fico_min: Optional[int] = Field(None, description="Minimum FICO")
    fico_max: Optional[int] = Field(None, description="Maximum FICO (null means unbounded)")
    dscr_min: Optional[float] = Field(None, description="Minimum DSCR")
    dscr_max: Optional[float] = Field(None, description="Maximum DSCR")
    loan_min: Optional[int] = Field(None, description="Minimum loan amount")
    loan_max: Optional[int] = Field(None, description="Maximum loan amount")
    
    # Additional dimensions
    property_type: Optional[PropertyType] = Field(None, description="Property type restriction")
    transaction_type: Optional[TransactionType] = Field(None, description="Transaction type")
    borrower_type: Optional[BorrowerType] = Field(None, description="Borrower type restriction")
    occupancy_type: Optional[OccupancyType] = Field(None, description="Occupancy type")
    state: Optional[str] = Field(None, description="State restriction")
    
    # Result values
    result_ltv: Optional[float] = Field(None, description="Resulting max LTV (%)")
    result_cltv: Optional[float] = Field(None, description="Resulting max CLTV (%)")
    result_rate_adjustment: Optional[float] = Field(None, description="Rate adjustment")
    result_eligible: bool = Field(True, description="Whether this combination is eligible")
    result_description: Optional[str] = Field(None, description="Additional result info")

    def matches(self, **conditions) -> bool:
        """Check if conditions match this matrix cell"""
        if "fico" in conditions:
            fico = conditions["fico"]
            if self.fico_min is not None and fico < self.fico_min:
                return False
            if self.fico_max is not None and fico > self.fico_max:
                return False
        
        if "dscr" in conditions:
            dscr = conditions["dscr"]
            if self.dscr_min is not None and dscr < self.dscr_min:
                return False
            if self.dscr_max is not None and dscr > self.dscr_max:
                return False
        
        if "loan_amount" in conditions:
            loan = conditions["loan_amount"]
            if self.loan_min is not None and loan < self.loan_min:
                return False
            if self.loan_max is not None and loan > self.loan_max:
                return False
        
        if "property_type" in conditions and self.property_type:
            if conditions["property_type"] != self.property_type:
                return False
        
        if "transaction_type" in conditions and self.transaction_type:
            if conditions["transaction_type"] != self.transaction_type:
                return False
        
        if "state" in conditions and self.state:
            if conditions["state"] != self.state:
                return False
        
        return True


class Matrix(GraphNode):
    """
    Eligibility matrix containing multiple cells.
    
    Relationships:
        (Product)-[:HAS_MATRIX]->(Matrix)
        (Matrix)-[:HAS_CELL]->(Cell)
    """
    name: str = Field(..., description="Matrix name")
    dimensions: List[str] = Field(default_factory=list, description="Dimension names (e.g., ['FICO', 'DSCR'])")
    purpose: str = Field(default="eligibility", description="Matrix purpose: eligibility, pricing, llpa_grid")
    description: Optional[str] = Field(None, description="Matrix description")
    cells: List[Cell] = Field(default_factory=list, description="Matrix cells (nested)")
    
    def add_cell(self, cell: Cell) -> None:
        """Add a cell to this matrix."""
        self.cells.append(cell)
    
    @property
    def cell_count(self) -> int:
        """Number of cells in this matrix."""
        return len(self.cells)


# =============================================================================
# Document Requirement Models
# =============================================================================


class DocumentRequirement(GraphNode):
    """
    Document requirement triggered by conditions.
    
    Examples:
    - IF loan_amount > 2000000 THEN second_appraisal required
    - IF Foreign National THEN passport and visa required
    """
    trigger_field: str = Field(..., description="Field that triggers this requirement")
    trigger_operator: OperatorType = Field(..., description="Trigger condition operator")
    trigger_value: Union[int, float, str, List[Union[int, float, str]]] = Field(
        ..., description="Trigger condition value"
    )
    required_documents: List[str] = Field(..., description="List of required documents")
    description: Optional[str] = Field(None, description="Requirement description")


# =============================================================================
# Product Models
# =============================================================================


class ARMParams(BaseModel):
    """ARM-specific parameters"""
    index: Optional[str] = Field(None, description="Rate index (e.g., 30-day average SOFR)")
    margin: Optional[float] = Field(None, description="Margin rate (%)")
    initial_cap: Optional[float] = Field(None, description="Initial adjustment cap (%)")
    periodic_cap: Optional[float] = Field(None, description="Periodic adjustment cap (%)")
    lifetime_cap: Optional[float] = Field(None, description="Lifetime cap (%)")
    adjustment_period: Optional[str] = Field(None, description="Adjustment period (e.g., 6 months)")


class Product(GraphNode):
    """
    Mortgage product definition.
    
    Examples:
    - 5/6 ARM SOFR
    - 30 Year Fixed
    """
    name: str = Field(..., description="Product name")
    type: ProductType = Field(..., description="Product type")
    description: Optional[str] = Field(None, description="Product description")
    arm_params: Optional[ARMParams] = Field(None, description="ARM parameters if applicable")
    fixed_term_years: Optional[int] = Field(None, description="Fixed term in years")
    qualify_rate: Optional[str] = Field(None, description="Rate to qualify at")


# =============================================================================
# Program Models
# =============================================================================


class Program(GraphNode):
    """
    Top-level mortgage program definition.
    
    Example: JMAC DSCR PRIME
    """
    name: str = Field(..., description="Program name")
    lender: str = Field(..., description="Lender name")
    version: str = Field(default="unknown", description="Program version")
    effective_date: str = Field(default="unknown", description="Effective date")
    status: ProgramStatus = Field(ProgramStatus.ACTIVE, description="Program status")
    description: Optional[str] = Field(None, description="Program description")
    allows_high_cost: bool = Field(False, description="High-cost loans allowed")
    max_points_fees: Optional[float] = Field(None, description="Max points and fees (%)")
    prepay_options: Optional[List[int]] = Field(None, description="Prepayment periods (years)")
    prepay_structure: Optional[str] = Field(None, description="Prepayment structure")
    escrow_required: Optional[bool] = Field(None, description="Escrow required")
    escrow_waiver_conditions: Optional[str] = Field(None, description="Conditions for escrow waiver")


# =============================================================================
# Extraction Models
# =============================================================================


class ExtractionStage(BaseModel):
    """Single extraction stage configuration"""
    name: str = Field(..., description="Stage name")
    target: str = Field(..., description="What to extract in this stage")
    token_budget: int = Field(..., description="Token budget for this stage")
    special_handling: Optional[str] = Field(None, description="Special handling type")


class FullExtractionResult(BaseModel):
    """Complete extraction result from all stages"""
    program: Optional[Program] = Field(None, description="Extracted program")
    products: List[Product] = Field(default_factory=list, description="Extracted products")
    rules: List[Rule] = Field(default_factory=list, description="Extracted rules")
    matrices: List[Matrix] = Field(default_factory=list, description="Eligibility matrices")
    document_requirements: List[DocumentRequirement] = Field(
        default_factory=list, description="Document requirements"
    )
    
    # Metadata
    source_file: Optional[str] = Field(None, description="Source file path")
    extraction_timestamp: Optional[str] = Field(None, description="Extraction timestamp")
    stages_completed: List[str] = Field(default_factory=list, description="Completed stages")
    errors: List[str] = Field(default_factory=list, description="Extraction errors")
    
    def is_valid(self) -> bool:
        """Check if extraction produced valid results"""
        return self.program is not None and len(self.products) > 0
    
    def to_graph_data(self) -> Dict[str, List[Any]]:
        """Convert to graph-importable data"""
        all_cells = []
        for matrix in self.matrices:
            all_cells.extend(matrix.cells)
        
        return {
            "programs": [self.program] if self.program else [],
            "products": self.products,
            "rules": self.rules,
            "matrices": self.matrices,
            "cells": all_cells,
            "document_requirements": self.document_requirements,
        }
    
    @property
    def cells(self) -> List[Cell]:
        """Get all cells from all matrices (flattened view)."""
        return [cell for matrix in self.matrices for cell in matrix.cells]
    
    @property
    def cell_count(self) -> int:
        """Total number of cells across all matrices."""
        return len(self.cells)
    
    @property
    def matrix_cells(self) -> List[Cell]:
        """Alias for cells (backward compatibility)"""
        return self.cells
    
    @property
    def state_overlays(self) -> List[Rule]:
        """Get rules with category='state' (backward compatibility)"""
        return [r for r in self.rules if r.category == "state"]
