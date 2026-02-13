"""
Eligibility Calculator - Precise parameter calculation for mortgage eligibility.

This module provides functionality to calculate precise eligibility parameters
(LTV, rate adjustments, required documents, etc.) for a specific program/product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from domain.mortgage.queries import (
    EligibilityResult,
    UserConditions,
)


@dataclass
class CalculationResult:
    """Detailed result of eligibility calculation"""
    eligible: bool
    program_id: str
    program_name: str
    
    # Calculated parameters
    max_ltv: Optional[float] = None
    max_cltv: Optional[float] = None
    rate_adjustment: float = 0.0
    
    # Matrix match info
    matched_cell_id: Optional[str] = None
    matched_dimensions: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    
    # Adjustments applied
    ltv_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    rate_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Requirements
    required_documents: List[str] = field(default_factory=list)
    required_reserves_months: Optional[int] = None
    
    # Restrictions
    restrictions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Rules for LLM reasoning
    state_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Detailed breakdown
    conditions_checked: Dict[str, bool] = field(default_factory=dict)
    conditions_failed: List[str] = field(default_factory=list)


class EligibilityCalculator:
    """
    Calculate precise eligibility parameters.
    
    This class provides detailed calculations including:
    1. Matrix cell matching
    2. LTV/CLTV calculation with adjustments
    3. Rate adjustment calculation
    4. Document requirement determination
    5. Reserve requirement calculation
    
    Example:
        >>> calc = EligibilityCalculator(storage)
        >>> result = calc.calculate("prog_123", conditions)
        >>> print(f"Max LTV: {result.max_ltv}%")
        >>> print(f"Required docs: {result.required_documents}")
    """
    
    def __init__(self, storage):
        self.storage = storage
    
    def calculate(
        self,
        program_id: str,
        conditions: UserConditions,
        product_id: str = None,
    ) -> CalculationResult:
        """
        Calculate eligibility parameters for a specific program.
        
        Args:
            program_id: Program ID
            conditions: User conditions
            product_id: Optional product ID (if known)
            
        Returns:
            CalculationResult with detailed parameters
        """
        # Initialize result
        result = CalculationResult(
            eligible=True,
            program_id=program_id,
            program_name=self._get_program_name(program_id),
        )
        
        # Step 1: Find matching matrix cell
        cell_result = self._find_matrix_cell(program_id, conditions, product_id)
        if cell_result:
            result.matched_cell_id = cell_result.get("cell_id")
            result.max_ltv = cell_result.get("result_ltv")
            result.max_cltv = cell_result.get("result_cltv")
            
            # Store matched dimensions
            result.matched_dimensions = {
                "fico": (cell_result.get("fico_min"), cell_result.get("fico_max")),
                "dscr": (cell_result.get("dscr_min"), cell_result.get("dscr_max")),
                "loan_amount": (cell_result.get("loan_min"), cell_result.get("loan_max")),
            }
        else:
            # No matching cell found
            result.eligible = False
            result.restrictions.append("No matching eligibility matrix cell found")
            return result
        
        # Step 2: Apply state overlays
        self._apply_state_overlays(result, conditions)
        
        # Step 3: Apply rule adjustments
        self._apply_rules(result, conditions)
        
        # Step 4: Calculate document requirements
        self._calculate_documents(result, conditions)
        
        # Step 5: Calculate reserve requirements
        self._calculate_reserves(result, conditions)
        
        # Step 6: Validate hard conditions
        self._validate_hard_conditions(result, conditions)
        
        return result
    
    def _get_program_name(self, program_id: str) -> str:
        """Get program name from ID."""
        query = "MATCH (p:Program {id: $id}) RETURN p.name as name"
        results = self.storage.execute_query(query, {"id": program_id})
        return results[0].get("name", "") if results else ""
    
    def _find_matrix_cell(
        self,
        program_id: str,
        conditions: UserConditions,
        product_id: str = None,
    ) -> Optional[Dict]:
        """Find the best matching matrix cell."""
        
        # Build product filter
        product_filter = ""
        if product_id:
            product_filter = "AND prod.id = $product_id"
        
        query = f"""
        MATCH (prog:Program {{id: $program_id}})-[:HAS_PRODUCT]->(prod:Product)
        {product_filter}
        OPTIONAL MATCH (prod)-[:HAS_MATRIX]->(m:Matrix)-[:HAS_CELL]->(cell:Cell)
        OPTIONAL MATCH (prod)-[:HAS_CELL]->(cell2:Cell)
        WITH prog, prod, COALESCE(cell, cell2) as cell
        WHERE cell IS NOT NULL
          AND cell.result_eligible = true
          AND ($fico IS NULL OR 
              (cell.fico_min IS NULL OR cell.fico_min <= $fico) AND
              (cell.fico_max IS NULL OR cell.fico_max >= $fico))
          AND ($dscr IS NULL OR
              (cell.dscr_min IS NULL OR cell.dscr_min <= $dscr) AND
              (cell.dscr_max IS NULL OR cell.dscr_max >= $dscr))
          AND ($loan_amount IS NULL OR
              (cell.loan_min IS NULL OR cell.loan_min <= $loan_amount) AND
              (cell.loan_max IS NULL OR cell.loan_max >= $loan_amount))
        RETURN cell.id as cell_id, cell.result_ltv as result_ltv,
               cell.result_cltv as result_cltv,
               cell.fico_min as fico_min, cell.fico_max as fico_max,
               cell.dscr_min as dscr_min, cell.dscr_max as dscr_max,
               cell.loan_min as loan_min, cell.loan_max as loan_max
        ORDER BY cell.result_ltv DESC
        LIMIT 1
        """
        
        params = {
            "program_id": program_id,
            "product_id": product_id,
            "fico": conditions.fico,
            "dscr": conditions.dscr,
            "loan_amount": conditions.loan_amount,
        }
        
        results = self.storage.execute_query(query, params)
        return results[0] if results else None
    
    def _apply_state_overlays(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Get state-specific rules for LLM-based reasoning."""
        if not conditions.state:
            return
        
        # Query rules with category='state' for this program
        query = """
        MATCH (prog:Program {id: $program_id})-[r]->(rule:Rule)
        WHERE rule.category = 'state'
        RETURN rule.trigger as trigger, rule.action as action, rule.excludes as excludes
        """
        
        rules = self.storage.execute_query(query, {
            "program_id": result.program_id,
        })
        
        # Store rules for later LLM processing
        if rules:
            result.state_rules = rules
    
    def _apply_rules(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Apply rule-based adjustments."""
        
        # Check for first-time investor rules
        if conditions.first_time_investor:
            self._apply_fti_rules(result, conditions)
        
        # Check for foreign national rules
        if conditions.borrower_type:
            self._apply_borrower_type_rules(result, conditions)
    
    def _apply_fti_rules(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Apply First Time Investor specific rules."""
        
        # Additional FTI validation
        if conditions.dscr and conditions.dscr <= 1.0:
            result.eligible = False
            result.restrictions.append("FTI requires DSCR > 1.0")
        
        if conditions.fico and conditions.fico < 700:
            result.eligible = False
            result.restrictions.append("FTI requires FICO >= 700")
    
    def _apply_borrower_type_rules(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Apply borrower type specific rules."""
        
        borrower_type = conditions.borrower_type.value if hasattr(
            conditions.borrower_type, 'value'
        ) else str(conditions.borrower_type)
        
        # Foreign National specific
        if borrower_type == "Foreign_National":
            result.required_documents.extend(["passport", "visa", "ACH_form"])
            if result.max_ltv and result.max_ltv > 75:
                result.ltv_adjustments.append({
                    "type": "borrower_type",
                    "original": result.max_ltv,
                    "adjusted": 75.0,
                    "reason": "Foreign National LTV cap",
                })
                result.max_ltv = 75.0
    
    def _calculate_documents(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Calculate required documents based on conditions."""
        
        # Base documents
        result.required_documents.extend([
            "loan_application",
            "credit_report",
            "appraisal",
        ])
        
        # Deduplicate
        result.required_documents = list(set(result.required_documents))
    
    def _calculate_reserves(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Calculate reserve requirements."""
        result.required_reserves_months = 6  # Default
    
    def _validate_hard_conditions(
        self,
        result: CalculationResult,
        conditions: UserConditions,
    ):
        """Validate hard rejection conditions."""
        pass  # Placeholder for additional validation
    
    def to_eligibility_result(self, calc_result: CalculationResult) -> EligibilityResult:
        """Convert CalculationResult to EligibilityResult."""
        return EligibilityResult(
            eligible=calc_result.eligible,
            program_id=calc_result.program_id,
            program_name=calc_result.program_name,
            max_ltv=calc_result.max_ltv,
            max_cltv=calc_result.max_cltv,
            rate_adjustment=calc_result.rate_adjustment,
            required_documents=calc_result.required_documents,
            restrictions=calc_result.restrictions,
            conditions_met=[k for k, v in calc_result.conditions_checked.items() if v],
            conditions_not_met=calc_result.conditions_failed,
            matrix_cell_id=calc_result.matched_cell_id,
        )
