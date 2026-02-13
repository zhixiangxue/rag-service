"""
Mortgage Graph Storage - Domain layer for storing and querying mortgage programs.

This module provides mortgage-specific operations on top of the base GraphStorage.
It implements the business logic for:
1. Storing complete program structures
2. Finding matching programs
3. Eligibility queries
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from zag.storages.graph import GraphStorage


class MortgageGraphStorage:
    """
    Mortgage-specific graph storage operations.
    
    This class wraps a base GraphStorage and adds mortgage domain logic.
    
    Example:
        >>> from zag.storages.graph import FalkorDBGraphStorage
        >>> base_storage = FalkorDBGraphStorage(host="localhost", port=6379)
        >>> storage = MortgageGraphStorage(base_storage)
        >>> storage.store_program(extraction_result.to_graph_data())
    """
    
    def __init__(self, storage: GraphStorage):
        """
        Initialize mortgage graph storage.
        
        Args:
            storage: Base GraphStorage instance
        """
        self.storage = storage
    
    def connect(self) -> bool:
        """Connect to the database."""
        return self.storage.connect()
    
    def disconnect(self) -> bool:
        """Disconnect from the database."""
        return self.storage.disconnect()
    
    def __enter__(self):
        self.storage.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.storage.__exit__(exc_type, exc_val, exc_tb)
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """Execute a raw Cypher query."""
        return self.storage.execute_query(query, params)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert object to dictionary with proper serialization."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return {k: self._serialize_value(v) for k, v in obj.items()}
        if hasattr(obj, 'model_dump'):
            return {k: self._serialize_value(v) for k, v in obj.model_dump(exclude_none=True).items()}
        return {}
    
    def _serialize_value(self, v: Any, is_list_element: bool = False) -> Any:
        """Serialize a value for storage.
        
        Args:
            v: Value to serialize
            is_list_element: True if this value is an element in a list
        """
        import json
        
        if v is None:
            return None
        # Check Enum BEFORE basic types (Enum can be subclass of str/int)
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, list):
            # List elements are marked so dicts inside lists stay as dicts
            return [self._serialize_value(item, is_list_element=True) for item in v]
        if isinstance(v, dict):
            # Dicts inside lists should stay as dicts (they'll be separate nodes)
            # Property values that are dicts must be JSON strings for FalkorDB
            if is_list_element:
                return {k: self._serialize_value(val) for k, val in v.items()}
            # All property dicts must be JSON strings
            return json.dumps({k: self._serialize_value(val) for k, val in v.items()})
        if hasattr(v, 'model_dump'):
            # Handle Pydantic models
            return self._serialize_value(v.model_dump(exclude_none=True), is_list_element)
        if hasattr(v, 'value'):
            # Handle objects with value attribute (non-Enum)
            return v.value
        return str(v)
    
    # =========================================================================
    # Mortgage Program Operations
    # =========================================================================
    
    def store_program(self, program_data: Dict[str, Any]) -> str:
        """
        Store a complete program with all related entities.
        
        Graph structure:
            (Program)-[:HAS_PRODUCT]->(Product)
            (Program)-[:HAS_RULE]->(Rule)
            (Program)-[:REQUIRES_DOCUMENT]->(DocumentRequirement)
            (Product)-[:HAS_MATRIX]->(Matrix)-[:HAS_CELL]->(Cell)
            (Product)-[:HAS_RULE]->(Rule)
        
        Args:
            program_data: Dictionary with 'programs', 'products', 'rules', etc.
                         Values can be Pydantic models or dicts.
            
        Returns:
            Program ID
        """
        # Create Program node (note: to_graph_data returns 'programs' list)
        programs = program_data.get("programs", [])
        if not programs:
            # Try singular key for backwards compatibility
            programs = [program_data.get("program")] if program_data.get("program") else []
        
        if not programs:
            raise ValueError("No program data provided")
        
        program = self._to_dict(programs[0])
        program_id = self.storage.create_node("Program", program)
        
        # Create Products
        product_ids = []
        for product in program_data.get("products", []):
            product_id = self.storage.create_node("Product", self._to_dict(product))
            product_ids.append(product_id)
            self.storage.create_relationship(
                "Program", program_id,
                "Product", product_id,
                "HAS_PRODUCT"
            )
        
        # Create Matrices with Cells
        matrix_product_map = {}
        first_product_id = product_ids[0] if product_ids else None
        
        for matrix in program_data.get("matrices", []):
            matrix_dict = self._to_dict(matrix)
            # Extract cells before storing matrix
            cells = matrix_dict.pop("cells", [])
            matrix_id = self.storage.create_node("Matrix", matrix_dict)
            # Link matrix to first product (default)
            if first_product_id:
                self.storage.create_relationship(
                    "Product", first_product_id,
                    "Matrix", matrix_id,
                    "HAS_MATRIX"
                )
                matrix_product_map[matrix_id] = first_product_id
            
            # Create cells for this matrix
            for cell in cells:
                cell_dict = self._to_dict(cell) if hasattr(cell, 'model_dump') else cell
                cell_id = self.storage.create_node("Cell", cell_dict)
                self.storage.create_relationship(
                    "Matrix", matrix_id,
                    "Cell", cell_id,
                    "HAS_CELL"
                )
        
        # Create Cells (standalone - backward compatibility)
        cells_data = program_data.get("cells", program_data.get("matrix_cells", []))
        first_matrix_id = list(matrix_product_map.keys())[0] if matrix_product_map else None
        
        for cell in cells_data:
            cell_dict = self._to_dict(cell)
            cell_id = self.storage.create_node("Cell", cell_dict)
            # Link to matrix if exists, otherwise to product
            if first_matrix_id:
                self.storage.create_relationship(
                    "Matrix", first_matrix_id,
                    "Cell", cell_id,
                    "HAS_CELL"
                )
            elif first_product_id:
                self.storage.create_relationship(
                    "Product", first_product_id,
                    "Cell", cell_id,
                    "HAS_CELL"
                )
        
        # Create Rules
        rules_data = program_data.get("rules", program_data.get("state_overlays", []))
        for rule in rules_data:
            rule_id = self.storage.create_node("Rule", self._to_dict(rule))
            # Link to program by default
            self.storage.create_relationship(
                "Program", program_id,
                "Rule", rule_id,
                "HAS_RULE"
            )
        
        # Create Document Requirements
        for req in program_data.get("document_requirements", []):
            req_id = self.storage.create_node("DocumentRequirement", self._to_dict(req))
            self.storage.create_relationship(
                "Program", program_id,
                "DocumentRequirement", req_id,
                "REQUIRES_DOCUMENT"
            )
        
        return program_id
    
    def find_matching_programs(
        self,
        user_conditions: Dict[str, Any],
    ) -> List[Dict]:
        """
        Find programs matching user conditions.
        
        Args:
            user_conditions: Dictionary with FICO, DSCR, loan_amount, state, etc.
            
        Returns:
            List of matching program info
        """
        fico = user_conditions.get("fico")
        dscr = user_conditions.get("dscr")
        loan_amount = user_conditions.get("loan_amount")
        state = user_conditions.get("state")
        
        # Build query to find matching cells
        query = """
        MATCH (prog:Program)-[:HAS_PRODUCT]->(prod:Product)
        OPTIONAL MATCH (prod)-[:HAS_MATRIX]->(m:Matrix)-[:HAS_CELL]->(cell:Cell)
        OPTIONAL MATCH (prod)-[:HAS_CELL]->(cell2:Cell)
        WITH prog, prod, COALESCE(cell, cell2) as cell
        WHERE cell IS NOT NULL
          AND ($fico IS NULL OR 
               (cell.fico_min IS NULL OR cell.fico_min <= $fico) AND
               (cell.fico_max IS NULL OR cell.fico_max >= $fico))
          AND ($dscr IS NULL OR
               (cell.dscr_min IS NULL OR cell.dscr_min <= $dscr) AND
               (cell.dscr_max IS NULL OR cell.dscr_max >= $dscr))
          AND ($loan_amount IS NULL OR
               (cell.loan_min IS NULL OR cell.loan_min <= $loan_amount) AND
               (cell.loan_max IS NULL OR cell.loan_max >= $loan_amount))
          AND cell.result_eligible = true
        RETURN prog.name as program, prog.lender as lender, 
               prod.name as product, cell.result_ltv as ltv, cell.id as cell_id
        """
        
        params = {
            "fico": fico,
            "dscr": dscr,
            "loan_amount": loan_amount,
        }
        
        results = self.storage.execute_query(query, params)
        
        # Filter by state rule (exclusion rules) - for LLM reasoning
        if state:
            filtered_results = []
            for r in results:
                # Get state rules for LLM to process
                rule_query = """
                MATCH (prog:Program {name: $program_name})-[:HAS_RULE]->(rule:Rule)
                WHERE rule.category = 'state'
                RETURN rule.trigger as trigger, rule.action as action, rule.excludes as excludes
                """
                rules = self.storage.execute_query(rule_query, {
                    "program_name": r["program"],
                })
                # Include state rules in result for LLM reasoning
                r["state_rules"] = rules
                filtered_results.append(r)
            results = filtered_results
        
        return results
    
    def get_program_by_id(self, program_id: str) -> Optional[Dict]:
        """Get a program by ID."""
        return self.storage.get_node("Program", program_id)
    
    def get_program_products(self, program_id: str) -> List[Dict]:
        """Get all products for a program."""
        query = """
        MATCH (prog:Program {id: $program_id})-[:HAS_PRODUCT]->(prod:Product)
        RETURN prod
        """
        results = self.storage.execute_query(query, {"program_id": program_id})
        return [r.get("prod", r) for r in results]
    
    def get_program_rules(self, program_id: str, category: str = None) -> List[Dict]:
        """Get rules for a program, optionally filtered by category."""
        if category:
            query = """
            MATCH (prog:Program {id: $program_id})-[:HAS_RULE]->(rule:Rule)
            WHERE rule.category = $category
            RETURN rule
            """
            params = {"program_id": program_id, "category": category}
        else:
            query = """
            MATCH (prog:Program {id: $program_id})-[:HAS_RULE]->(rule:Rule)
            RETURN rule
            """
            params = {"program_id": program_id}
        
        results = self.storage.execute_query(query, params)
        return [r.get("rule", r) for r in results]
    
    def list_programs(self, limit: int = 100) -> List[Dict]:
        """List all programs."""
        query = """
        MATCH (prog:Program)
        RETURN prog.id as id, prog.name as name, prog.lender as lender, prog.status as status
        LIMIT $limit
        """
        return self.storage.execute_query(query, {"limit": limit})
