"""
Program Matcher - Eligibility matching for mortgage programs.

This module provides functionality to match user conditions against
mortgage programs stored in the graph database.

Key Features:
1. Multi-dimensional matrix matching (FICO, DSCR, Loan Amount)
2. State overlay filtering
3. Condition validation
4. Match scoring and ranking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .graph import (
    BorrowerType,
    OccupancyType,
    PropertyType,
    TransactionType,
)
from .queries import (
    EligibilityResult,
    UserConditions,
)


@dataclass
class MatchResult:
    """Result of a single program match check"""
    program_id: str
    program_name: str
    product_id: str
    product_name: str
    eligible: bool
    max_ltv: Optional[float] = None
    max_cltv: Optional[float] = None
    matrix_cell_id: Optional[str] = None
    conditions_met: List[str] = field(default_factory=list)
    conditions_not_met: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    required_documents: List[str] = field(default_factory=list)
    state_overlay_applied: bool = False
    state_rules: List[Dict[str, Any]] = field(default_factory=list)  # For LLM reasoning
    match_score: float = 0.0


class ProgramMatcher:
    """
    Match user conditions against mortgage programs.
    
    This class implements the eligibility matching logic:
    1. Find programs with matching matrix cells
    2. Apply state overlay restrictions
    3. Validate additional conditions
    4. Calculate match score
    5. Rank results
    
    Example:
        >>> matcher = ProgramMatcher(storage)
        >>> conditions = UserConditions(fico=730, dscr=1.1, loan_amount=800000, state="CA")
        >>> results = matcher.match(conditions)
        >>> for r in results:
        ...     print(f"{r.program_name}: eligible={r.eligible}, ltv={r.max_ltv}")
    """
    
    def __init__(self, storage):
        """
        Initialize the matcher.
        
        Args:
            storage: GraphStorage instance for database access
        """
        self.storage = storage
    
    def match(
        self,
        conditions: UserConditions,
        limit: int = 10,
    ) -> List[EligibilityResult]:
        """
        Find matching programs for user conditions.
        
        Args:
            conditions: User input conditions
            limit: Maximum number of results
            
        Returns:
            List of EligibilityResult sorted by match score
        """
        # Step 1: Find matching matrix cells
        match_results = self._find_matching_cells(conditions)
        
        # Step 2: Apply state overlays
        match_results = self._apply_state_overlays(match_results, conditions)
        
        # Step 3: Validate conditions
        match_results = self._validate_conditions(match_results, conditions)
        
        # Step 4: Get required documents
        match_results = self._get_required_documents(match_results, conditions)
        
        # Step 5: Calculate match scores
        match_results = self._calculate_scores(match_results, conditions)
        
        # Step 6: Convert to EligibilityResult
        results = [self._to_eligibility_result(r) for r in match_results]
        
        # Step 7: Sort by score and limit
        results.sort(key=lambda x: (-int(x.eligible), -(x.max_ltv or 0), -x.match_score))
        
        return results[:limit]
    
    def _find_matching_cells(self, conditions: UserConditions) -> List[MatchResult]:
        """Find matrix cells matching user conditions."""
        # Query supports both new (Product->Matrix->Cell) and old (Product->Cell) structures
        query = """
        MATCH (prog:Program)-[:HAS_PRODUCT]->(prod:Product)
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
        RETURN prog.id as program_id, prog.name as program_name,
               prod.id as product_id, prod.name as product_name,
               cell.id as cell_id, cell.result_ltv as ltv, 
               cell.result_cltv as cltv
        """
        
        params = {
            "fico": conditions.fico,
            "dscr": conditions.dscr,
            "loan_amount": conditions.loan_amount,
        }
        
        results = self.storage.execute_query(query, params)
        
        match_results = []
        seen = set()
        
        for r in results:
            key = (r.get("program_id"), r.get("product_id"))
            if key in seen:
                continue
            seen.add(key)
            
            match_results.append(MatchResult(
                program_id=r.get("program_id", ""),
                program_name=r.get("program_name", ""),
                product_id=r.get("product_id", ""),
                product_name=r.get("product_name", ""),
                eligible=True,
                max_ltv=r.get("ltv"),
                max_cltv=r.get("cltv"),
                matrix_cell_id=r.get("cell_id"),
            ))
        
        return match_results
    
    def _apply_state_overlays(
        self,
        results: List[MatchResult],
        conditions: UserConditions,
    ) -> List[MatchResult]:
        """Get state-specific rules for LLM-based reasoning."""
        if not conditions.state:
            return results
        
        for result in results:
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
                result.state_overlay_applied = True
                result.state_rules = rules  # Will be used by LLM for reasoning
        
        return results
    
    def _validate_conditions(
        self,
        results: List[MatchResult],
        conditions: UserConditions,
    ) -> List[MatchResult]:
        """Validate additional conditions."""
        
        for result in results:
            # Query conditions for this program
            query = """
            MATCH (prog:Program {id: $program_id})-[:HAS_CONDITION]->(c:Condition)
            RETURN c.field as field, c.operator as op, c.value as value,
                   c.value_type as type, c.description as desc
            """
            
            condition_results = self.storage.execute_query(query, {
                "program_id": result.program_id,
            })
            
            for cond in condition_results:
                field = cond.get("field")
                op = cond.get("op")
                value = cond.get("value")
                desc = cond.get("desc", f"{field} {op} {value}")
                
                # Get user value
                user_value = getattr(conditions, field.lower(), None)
                if user_value is None:
                    user_value = conditions.model_dump().get(field.lower())
                
                if user_value is None:
                    # Condition not provided by user
                    result.conditions_not_met.append(desc)
                    continue
                
                # Validate condition
                if self._check_condition(user_value, op, value):
                    result.conditions_met.append(desc)
                else:
                    result.conditions_not_met.append(desc)
                    # For hard conditions, this could make ineligible
        
        return results
    
    def _check_condition(
        self,
        user_value: Any,
        operator: str,
        condition_value: Any,
    ) -> bool:
        """Check if user value satisfies the condition."""
        try:
            if operator == ">=":
                return float(user_value) >= float(condition_value)
            elif operator == "<=":
                return float(user_value) <= float(condition_value)
            elif operator == ">":
                return float(user_value) > float(condition_value)
            elif operator == "<":
                return float(user_value) < float(condition_value)
            elif operator == "=" or operator == "==":
                return user_value == condition_value
            elif operator == "!=":
                return user_value != condition_value
            elif operator == "in":
                if isinstance(condition_value, list):
                    return user_value in condition_value
                return str(user_value) in str(condition_value)
            elif operator == "between":
                if isinstance(condition_value, list) and len(condition_value) == 2:
                    return float(condition_value[0]) <= float(user_value) <= float(condition_value[1])
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _get_required_documents(
        self,
        results: List[MatchResult],
        conditions: UserConditions,
    ) -> List[MatchResult]:
        """Get required documents based on conditions."""
        
        for result in results:
            # Query document requirements
            query = """
            MATCH (prog:Program {id: $program_id})-[:REQUIRES_DOCUMENT]->(dr:DocumentRequirement)
            RETURN dr.trigger_field as field, dr.trigger_operator as op,
                   dr.trigger_value as value, dr.required_documents as docs
            """
            
            req_results = self.storage.execute_query(query, {
                "program_id": result.program_id,
            })
            
            for req in req_results:
                field = req.get("field")
                op = req.get("op")
                value = req.get("value")
                docs = req.get("docs", [])
                
                # Get user value for trigger field
                user_value = getattr(conditions, field.lower(), None)
                if user_value is None:
                    user_value = conditions.model_dump().get(field.lower())
                
                if user_value is not None:
                    if self._check_condition(user_value, op, value):
                        result.required_documents.extend(docs)
        
        # Deduplicate
        for result in results:
            result.required_documents = list(set(result.required_documents))
        
        return results
    
    def _calculate_scores(
        self,
        results: List[MatchResult],
        conditions: UserConditions,
    ) -> List[MatchResult]:
        """Calculate match scores for ranking."""
        
        for result in results:
            score = 0.0
            
            # Base score for eligibility
            if result.eligible:
                score += 0.5
            
            # LTV score (higher LTV = better)
            if result.max_ltv:
                score += min(result.max_ltv / 100.0, 0.3)
            
            # Condition match score
            total_conditions = len(result.conditions_met) + len(result.conditions_not_met)
            if total_conditions > 0:
                score += 0.1 * (len(result.conditions_met) / total_conditions)
            
            # Penalty for restrictions
            score -= 0.05 * len(result.restrictions)
            
            # Penalty for required documents
            score -= 0.02 * len(result.required_documents)
            
            result.match_score = max(0.0, min(1.0, score))
        
        return results
    
    def _to_eligibility_result(self, match: MatchResult) -> EligibilityResult:
        """Convert MatchResult to EligibilityResult."""
        return EligibilityResult(
            eligible=match.eligible,
            program_id=match.program_id,
            program_name=match.program_name,
            product_id=match.product_id,
            product_name=match.product_name,
            max_ltv=match.max_ltv,
            max_cltv=match.max_cltv,
            required_documents=match.required_documents,
            restrictions=match.restrictions,
            conditions_met=match.conditions_met,
            conditions_not_met=match.conditions_not_met,
            matrix_cell_id=match.matrix_cell_id,
            match_score=match.match_score,
        )
    
    def check_eligibility(
        self,
        program_id: str,
        conditions: UserConditions,
    ) -> EligibilityResult:
        """
        Check eligibility for a specific program.
        
        Args:
            program_id: Program ID to check
            conditions: User conditions
            
        Returns:
            Single EligibilityResult
        """
        # Create modified conditions with program filter
        results = self.match(conditions, limit=100)
        
        # Filter to specific program
        for r in results:
            if r.program_id == program_id:
                return r
        
        # Not found
        return EligibilityResult(
            eligible=False,
            program_id=program_id,
            program_name="",
            restrictions=["Program not found or no matching products"],
        )
