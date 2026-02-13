"""
Mortgage query models for eligibility checks.
"""

from typing import List, Optional, Set

from pydantic import BaseModel, Field

from .graph import (
    BorrowerType,
    EntityType,
    OccupancyType,
    PropertyType,
    TransactionType,
)


class UserConditions(BaseModel):
    """
    User input conditions for eligibility queries.
    """
    # Borrower info
    fico: Optional[int] = Field(None, description="Credit score")
    dscr: Optional[float] = Field(None, description="Debt Service Coverage Ratio")
    dti: Optional[float] = Field(None, description="Debt-to-Income ratio")
    borrower_type: Optional[BorrowerType] = Field(None, description="Borrower type")
    first_time_investor: bool = Field(False, description="First time investor flag")
    
    # Property info
    property_type: Optional[PropertyType] = Field(None, description="Property type")
    occupancy_type: Optional[OccupancyType] = Field(None, description="Occupancy type")
    property_value: Optional[int] = Field(None, description="Property value")
    state: Optional[str] = Field(None, description="Property state")
    
    # Loan info
    loan_amount: Optional[int] = Field(None, description="Loan amount")
    transaction_type: Optional[TransactionType] = Field(None, description="Transaction type")
    ltv_requested: Optional[float] = Field(None, description="Requested LTV")
    
    # Entity info
    entity_type: Optional[EntityType] = Field(None, description="Entity type for vesting")
    
    # Additional context
    reserves_months: Optional[int] = Field(None, description="Reserves in months")
    existing_docs: Optional[Set[str]] = Field(None, description="Existing documents")


class EligibilityResult(BaseModel):
    """
    Result of eligibility check.
    """
    eligible: bool = Field(..., description="Whether eligible")
    program_id: str = Field(..., description="Program ID")
    program_name: str = Field(..., description="Program name")
    product_id: Optional[str] = Field(None, description="Product ID")
    product_name: Optional[str] = Field(None, description="Product name")
    
    # Calculated parameters
    max_ltv: Optional[float] = Field(None, description="Maximum LTV")
    max_cltv: Optional[float] = Field(None, description="Maximum CLTV")
    rate_adjustment: Optional[float] = Field(None, description="Rate adjustment")
    
    # Requirements and restrictions
    required_documents: List[str] = Field(default_factory=list, description="Required documents")
    restrictions: List[str] = Field(default_factory=list, description="Active restrictions")
    conditions_met: List[str] = Field(default_factory=list, description="Conditions met")
    conditions_not_met: List[str] = Field(default_factory=list, description="Conditions not met")
    
    # Match details
    matrix_cell_id: Optional[str] = Field(None, description="Matched matrix cell ID")
    match_score: Optional[float] = Field(None, description="Match score (0-1)")


class ProgramComparison(BaseModel):
    """
    Comparison result across multiple programs.
    """
    user_conditions: UserConditions = Field(..., description="Input conditions")
    results: List[EligibilityResult] = Field(..., description="Eligibility results")
    
    def get_eligible(self) -> List[EligibilityResult]:
        """Get only eligible results"""
        return [r for r in self.results if r.eligible]
    
    def get_best_ltv(self) -> Optional[EligibilityResult]:
        """Get result with highest LTV"""
        eligible = self.get_eligible()
        if not eligible:
            return None
        return max(eligible, key=lambda x: x.max_ltv or 0)
    
    def sort_by_eligibility(self) -> List[EligibilityResult]:
        """Sort results by eligibility and LTV"""
        return sorted(
            self.results,
            key=lambda x: (-int(x.eligible), -(x.max_ltv or 0), len(x.restrictions))
        )
