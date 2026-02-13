"""
Graph API endpoints for mortgage program queries.

This module provides REST API endpoints for:
- Listing programs
- Checking eligibility
- Finding matching programs
- Getting program details
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from ..schemas import ApiResponse
from .. import config
from domain.mortgage import (
    MortgageGraphStorage,
    ProgramMatcher,
    MatchResult,
    UserConditions,
    EligibilityResult,
    BorrowerType,
    PropertyType,
    OccupancyType,
    TransactionType,
)

router = APIRouter(prefix="/graph", tags=["graph"])


# Request/Response schemas
class EligibilityRequest(BaseModel):
    """Request for eligibility check."""
    fico: Optional[int] = Field(None, description="Credit score")
    dscr: Optional[float] = Field(None, description="Debt Service Coverage Ratio")
    dti: Optional[float] = Field(None, description="Debt-to-Income ratio")
    borrower_type: Optional[str] = Field(None, description="Borrower type")
    first_time_investor: bool = Field(False, description="First time investor flag")
    property_type: Optional[str] = Field(None, description="Property type")
    occupancy_type: Optional[str] = Field(None, description="Occupancy type")
    property_value: Optional[int] = Field(None, description="Property value")
    state: Optional[str] = Field(None, description="Property state")
    loan_amount: Optional[int] = Field(None, description="Loan amount")
    transaction_type: Optional[str] = Field(None, description="Transaction type")
    ltv_requested: Optional[float] = Field(None, description="Requested LTV")
    
    def to_user_conditions(self) -> UserConditions:
        """Convert to UserConditions model."""
        # Parse enum values
        borrower_type = None
        if self.borrower_type:
            try:
                borrower_type = BorrowerType(self.borrower_type)
            except ValueError:
                pass
        
        property_type = None
        if self.property_type:
            try:
                property_type = PropertyType(self.property_type)
            except ValueError:
                pass
        
        occupancy_type = None
        if self.occupancy_type:
            try:
                occupancy_type = OccupancyType(self.occupancy_type)
            except ValueError:
                pass
        
        transaction_type = None
        if self.transaction_type:
            try:
                transaction_type = TransactionType(self.transaction_type)
            except ValueError:
                pass
        
        return UserConditions(
            fico=self.fico,
            dscr=self.dscr,
            dti=self.dti,
            borrower_type=borrower_type,
            first_time_investor=self.first_time_investor,
            property_type=property_type,
            occupancy_type=occupancy_type,
            property_value=self.property_value,
            state=self.state,
            loan_amount=self.loan_amount,
            transaction_type=transaction_type,
            ltv_requested=self.ltv_requested,
        )


class ProgramResponse(BaseModel):
    """Response for a single program."""
    id: str
    name: str
    lender: str
    status: str = "active"


class EligibilityResponse(BaseModel):
    """Response for eligibility check."""
    eligible: bool
    program_id: str
    program_name: str
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    max_ltv: Optional[float] = None
    max_cltv: Optional[float] = None
    rate_adjustment: Optional[float] = None
    required_documents: List[str] = []
    restrictions: List[str] = []
    conditions_met: List[str] = []
    conditions_not_met: List[str] = []
    matrix_cell_id: Optional[str] = None
    match_score: Optional[float] = None
    state_rules: List[Dict[str, Any]] = []


def _get_storage() -> MortgageGraphStorage:
    """Get mortgage graph storage instance."""
    from zag.storages.graph import FalkorDBGraphStorage
    
    base_storage = FalkorDBGraphStorage(
        host=config.FALKORDB_HOST,
        port=config.FALKORDB_PORT,
        graph_name="mortgage_programs",
    )
    return MortgageGraphStorage(base_storage)


@router.get("/programs", response_model=ApiResponse[List[ProgramResponse]])
async def list_programs(limit: int = 100):
    """List all mortgage programs.
    
    Args:
        limit: Maximum number of programs to return
        
    Returns:
        List of programs with basic info
    """
    try:
        storage = _get_storage()
        with storage.storage:
            programs = storage.list_programs(limit=limit)
            
            return ApiResponse(
                success=True,
                code=200,
                data=[
                    ProgramResponse(
                        id=p.get("id", ""),
                        name=p.get("name", ""),
                        lender=p.get("lender", ""),
                        status=p.get("status", "active"),
                    )
                    for p in programs
                ]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list programs: {str(e)}")


@router.get("/programs/{program_id}", response_model=ApiResponse[Dict[str, Any]])
async def get_program(program_id: str):
    """Get program details by ID.
    
    Args:
        program_id: Program ID
        
    Returns:
        Program details including products and rules
    """
    try:
        storage = _get_storage()
        with storage.storage:
            program = storage.get_program_by_id(program_id)
            if not program:
                raise HTTPException(status_code=404, detail="Program not found")
            
            # Get related data
            products = storage.get_program_products(program_id)
            rules = storage.get_program_rules(program_id)
            
            return ApiResponse(
                success=True,
                code=200,
                data={
                    "program": program,
                    "products": products,
                    "rules": rules,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get program: {str(e)}")


@router.post("/eligibility", response_model=ApiResponse[List[EligibilityResponse]])
async def check_eligibility(request: EligibilityRequest, limit: int = 10):
    """Check eligibility across all programs.
    
    This endpoint finds all programs that match the user's conditions
    and returns eligibility results with LTV, restrictions, etc.
    
    Args:
        request: User conditions for eligibility check
        limit: Maximum number of results
        
    Returns:
        List of eligibility results sorted by match score
    """
    try:
        storage = _get_storage()
        with storage.storage:
            # Create matcher
            matcher = ProgramMatcher(storage.storage)
            
            # Convert request to UserConditions
            conditions = request.to_user_conditions()
            
            # Find matching programs
            results = matcher.match(conditions, limit=limit)
            
            # Get state rules for each result
            response_data = []
            for r in results:
                # Get state rules if state is provided
                state_rules = []
                if request.state:
                    rules = storage.get_program_rules(r.program_id, category="state")
                    state_rules = rules
                
                response_data.append(EligibilityResponse(
                    eligible=r.eligible,
                    program_id=r.program_id,
                    program_name=r.program_name,
                    product_id=r.product_id,
                    product_name=r.product_name,
                    max_ltv=r.max_ltv,
                    max_cltv=r.max_cltv,
                    required_documents=r.required_documents,
                    restrictions=r.restrictions,
                    conditions_met=r.conditions_met,
                    conditions_not_met=r.conditions_not_met,
                    matrix_cell_id=r.matrix_cell_id,
                    match_score=r.match_score,
                    state_rules=state_rules,
                ))
            
            return ApiResponse(
                success=True,
                code=200,
                data=response_data
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eligibility check failed: {str(e)}")


@router.post("/programs/{program_id}/eligibility", response_model=ApiResponse[EligibilityResponse])
async def check_program_eligibility(program_id: str, request: EligibilityRequest):
    """Check eligibility for a specific program.
    
    Args:
        program_id: Program ID to check
        request: User conditions for eligibility check
        
    Returns:
        Eligibility result for the specific program
    """
    try:
        storage = _get_storage()
        with storage.storage:
            # Create matcher
            matcher = ProgramMatcher(storage.storage)
            
            # Convert request to UserConditions
            conditions = request.to_user_conditions()
            
            # Check eligibility for specific program
            result = matcher.check_eligibility(program_id, conditions)
            
            # Get state rules if state is provided
            state_rules = []
            if request.state:
                state_rules = storage.get_program_rules(program_id, category="state")
            
            return ApiResponse(
                success=True,
                code=200,
                data=EligibilityResponse(
                    eligible=result.eligible,
                    program_id=result.program_id,
                    program_name=result.program_name,
                    product_id=result.product_id,
                    product_name=result.product_name,
                    max_ltv=result.max_ltv,
                    max_cltv=result.max_cltv,
                    required_documents=result.required_documents,
                    restrictions=result.restrictions,
                    conditions_met=result.conditions_met,
                    conditions_not_met=result.conditions_not_met,
                    matrix_cell_id=result.matrix_cell_id,
                    match_score=result.match_score,
                    state_rules=state_rules,
                )
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eligibility check failed: {str(e)}")


@router.get("/programs/{program_id}/products", response_model=ApiResponse[List[Dict[str, Any]]])
async def get_program_products(program_id: str):
    """Get products for a program.
    
    Args:
        program_id: Program ID
        
    Returns:
        List of products for the program
    """
    try:
        storage = _get_storage()
        with storage.storage:
            products = storage.get_program_products(program_id)
            return ApiResponse(
                success=True,
                code=200,
                data=products
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get products: {str(e)}")


@router.get("/programs/{program_id}/rules", response_model=ApiResponse[List[Dict[str, Any]]])
async def get_program_rules(program_id: str, category: str = None):
    """Get rules for a program.
    
    Args:
        program_id: Program ID
        category: Optional category filter (state, borrower, loan, property, transaction, general)
        
    Returns:
        List of rules for the program
    """
    try:
        storage = _get_storage()
        with storage.storage:
            rules = storage.get_program_rules(program_id, category=category)
            return ApiResponse(
                success=True,
                code=200,
                data=rules
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")
