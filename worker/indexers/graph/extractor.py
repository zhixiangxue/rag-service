"""
Mortgage Program Extractor - Multi-stage extraction for complex mortgage documents.

This module implements a staged extraction strategy to handle the complexity of
mortgage program documents. Instead of extracting everything in one pass, it
breaks down extraction into focused stages:

1. Program Overview: Basic program info (name, lender, version)
2. Product List: Product names, types, parameters
3. Eligibility Matrix: DSCR/LTV matrices (special table handling)
4. Borrower Restrictions: Foreign National, ITIN, First Time Investor rules
5. State Overlays: State-specific restrictions
6. Document Requirements: Documents required per scenario
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from domain.mortgage.graph import (
    ARMParams,
    Cell,
    DocumentRequirement,
    Matrix,
    ExtractionStage,
    FullExtractionResult,
    OperatorType,
    Product,
    ProductType,
    Program,
    ProgramStatus,
    Rule,
)


# =============================================================================
# Stage Schemas (for LLM structured output)
# =============================================================================


class ProgramOverviewSchema(BaseModel):
    """Schema for stage 1: Program overview"""
    program_name: str = Field(..., description="Program name (e.g., 'DSCR PRIME')")
    lender: str = Field(..., description="Lender name (e.g., 'JMAC Lending')")
    version: Optional[str] = Field(None, description="Version or effective date")
    description: Optional[str] = Field(None, description="Brief program description")
    allows_high_cost: Optional[bool] = Field(None, description="Whether high-cost loans are allowed")
    max_points_fees: Optional[float] = Field(None, description="Maximum points and fees percentage")
    prepay_options: Optional[List[int]] = Field(None, description="Prepayment penalty options in years")
    prepay_structure: Optional[str] = Field(None, description="Prepayment penalty structure description")


class ARMParamsSchema(BaseModel):
    """ARM parameters for product"""
    index: Optional[str] = Field(None, description="Rate index")
    margin: Optional[float] = Field(None, description="Margin percentage")
    initial_cap: Optional[float] = Field(None, description="Initial adjustment cap")
    periodic_cap: Optional[float] = Field(None, description="Periodic adjustment cap")
    lifetime_cap: Optional[float] = Field(None, description="Lifetime cap")
    adjustment_period: Optional[str] = Field(None, description="Adjustment period")


class ProductItemSchema(BaseModel):
    """Single product item"""
    name: str = Field(..., description="Product name (e.g., '5/6 ARM', '30 Year Fixed')")
    type: str = Field(..., description="Product type: Fixed, ARM, SOFR_ARM, etc.")
    description: Optional[str] = Field(None, description="Product description")
    arm_params: Optional[ARMParamsSchema] = Field(None, description="ARM parameters if applicable")
    fixed_term_years: Optional[int] = Field(None, description="Fixed term in years")
    qualify_rate: Optional[str] = Field(None, description="Rate to qualify at")


class ProductListSchema(BaseModel):
    """Schema for stage 2: Product list"""
    products: List[ProductItemSchema] = Field(default_factory=list)


class MatrixCellSchema(BaseModel):
    """Single matrix cell"""
    fico_min: Optional[int] = Field(None, description="Minimum FICO score")
    fico_max: Optional[int] = Field(None, description="Maximum FICO score (null if unbounded)")
    dscr_min: Optional[float] = Field(None, description="Minimum DSCR")
    dscr_max: Optional[float] = Field(None, description="Maximum DSCR")
    loan_min: Optional[int] = Field(None, description="Minimum loan amount")
    loan_max: Optional[int] = Field(None, description="Maximum loan amount")
    property_type: Optional[str] = Field(None, description="Property type restriction")
    transaction_type: Optional[str] = Field(None, description="Transaction type (Purchase, Cash-Out, etc.)")
    result_ltv: Optional[float] = Field(None, description="Resulting max LTV percentage")
    result_cltv: Optional[float] = Field(None, description="Resulting max CLTV percentage")
    result_eligible: bool = Field(True, description="Whether this combination is eligible")


class EligibilityMatrixSchema(BaseModel):
    """Schema for stage 3: Eligibility matrix"""
    matrix_name: Optional[str] = Field(None, description="Matrix name")
    dimensions: List[str] = Field(default_factory=list, description="Dimensions used (FICO, DSCR, LoanAmount, etc.)")
    cells: List[MatrixCellSchema] = Field(default_factory=list, description="Matrix cells")


class BorrowerRestrictionSchema(BaseModel):
    """Single borrower restriction"""
    borrower_type: str = Field(..., description="Borrower type (Foreign National, ITIN, First Time Investor, etc.)")
    restrictions: List[str] = Field(default_factory=list, description="List of restrictions")


class BorrowerRestrictionsSchema(BaseModel):
    """Schema for stage 5: Borrower restrictions"""
    restrictions: List[BorrowerRestrictionSchema] = Field(default_factory=list)


class StateOverlaySchema(BaseModel):
    """Single state overlay (deprecated - use RuleSchema)"""
    state: str = Field(..., description="State code (CA, TX, FL, etc.)")
    restriction_type: str = Field(..., description="Type of restriction")
    restriction_value: Optional[Any] = Field(None, description="Restriction value")
    restriction_description: str = Field(..., description="Human-readable description")
    excludes: bool = Field(False, description="Whether this excludes eligibility")


class RuleSchema(BaseModel):
    """Schema for rules (IFTTT-style for LLM extraction)"""
    category: str = Field(default="general", description="Rule category: state, borrower, loan, property, transaction, general")
    trigger: str = Field(..., description="Natural language trigger condition (IF part)")
    action: str = Field(..., description="Natural language action/constraint (THEN part)")
    excludes: bool = Field(False, description="Whether this excludes eligibility")


class StateOverlaysSchema(BaseModel):
    """Schema for stage 6: State overlays"""
    overlays: List[StateOverlaySchema] = Field(default_factory=list)


class DocRequirementSchema(BaseModel):
    """Single document requirement"""
    trigger_field: str = Field(..., description="Field that triggers this requirement")
    trigger_operator: str = Field(..., description="Trigger condition operator")
    trigger_value: Any = Field(..., description="Trigger condition value")
    required_documents: List[str] = Field(default_factory=list, description="Required documents")
    description: Optional[str] = Field(None, description="Requirement description")


class DocRequirementsSchema(BaseModel):
    """Schema for stage 7: Document requirements"""
    requirements: List[DocRequirementSchema] = Field(default_factory=list)


class GapItemSchema(BaseModel):
    """A single identified gap in the extraction"""
    stage: str = Field(
        ...,
        description=(
            "Which stage this gap belongs to: "
            "eligibility_matrix, state_overlays, borrower_restrictions, document_requirements"
        ),
    )
    description: str = Field(..., description="What is missing or incorrect")
    source_text: str = Field(
        ..., description="The exact text from the document that contains this information"
    )


class GapDetectionSchema(BaseModel):
    """Schema for gap detection verification pass"""
    is_complete: bool = Field(
        ...,
        description="True if the extraction looks complete and accurate, False if gaps were found",
    )
    gaps: List[GapItemSchema] = Field(
        default_factory=list, description="List of identified gaps"
    )


# =============================================================================
# Extraction Stages Configuration
# =============================================================================


# Default extraction stages
DEFAULT_EXTRACTION_STAGES = [
    ExtractionStage(
        name="program_overview",
        target="Program name, lender, effective date, prepayment options",
        token_budget=2000,
    ),
    ExtractionStage(
        name="product_list",
        target="Product names, types, ARM parameters, fixed terms",
        token_budget=3000,
    ),
    ExtractionStage(
        name="eligibility_matrix",
        target="DSCR/LTV eligibility matrices - focus on tables and decision matrices",
        token_budget=3000,  # Reduced to avoid token limits
        special_handling="table_extraction",
    ),
    ExtractionStage(
        name="borrower_restrictions",
        target="Foreign National, ITIN, First Time Investor, DACA restrictions",
        token_budget=5000,
    ),
    ExtractionStage(
        name="state_overlays",
        target="State-specific restrictions (IL, FL, CA, TX, etc.)",
        token_budget=4000,
    ),
    ExtractionStage(
        name="document_requirements",
        target="Document requirements triggered by conditions",
        token_budget=4000,
    ),
]


# =============================================================================
# Main Extractor Class
# =============================================================================


class MortgageProgramExtractor:
    """
    Multi-stage extractor for mortgage program documents.
    
    This extractor uses a staged approach to handle complex mortgage documents:
    1. Each stage focuses on a specific aspect of the program
    2. Results are combined into a complete FullExtractionResult
    3. Supports special handling for tables (matrices)
    
    Example:
        >>> extractor = MortgageProgramExtractor(
        ...     llm_uri="openai/gpt-4o",
        ...     api_key=os.getenv("OPENAI_API_KEY")
        ... )
        >>> result = await extractor.extract(document_content)
        >>> print(result.program.name)
        >>> print(len(result.products))
    """
    
    def __init__(
        self,
        llm_uri: str = "openai/gpt-4o",
        api_key: str = None,
        stages: List[ExtractionStage] = None,
        max_content_length: int = 15000,
    ):
        """
        Initialize the extractor.
        
        Args:
            llm_uri: LLM URI (e.g., "openai/gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            stages: Custom extraction stages (defaults to DEFAULT_EXTRACTION_STAGES)
            max_content_length: Maximum content length per extraction
        """
        self.llm_uri = llm_uri
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.stages = stages or DEFAULT_EXTRACTION_STAGES
        self.max_content_length = max_content_length
        
        # Initialize chak
        try:
            import chak
            self._chak = chak
        except ImportError:
            raise ImportError(
                "chak is required for MortgageProgramExtractor. "
                "Install it with: pip install chakpy"
            )
        
        self._conv = self._chak.Conversation(
            llm_uri,
            api_key=self.api_key,
        )
    
    async def extract(
        self,
        document_content: str,
        source_file: str = None,
        max_verification_rounds: int = 1,
    ) -> FullExtractionResult:
        """
        Extract complete program structure from document content.

        Args:
            document_content: Full document text
            source_file: Optional source file path for metadata
            max_verification_rounds: Number of gap-detection + fill rounds after initial extraction.
                Set to 0 to skip verification. Default is 1.

        Returns:
            FullExtractionResult with all extracted components
        """
        result = FullExtractionResult(
            source_file=source_file,
            extraction_timestamp=datetime.now().isoformat(),
        )

        # Use full document - no truncation
        content = document_content

        # Run each extraction stage
        for stage in self.stages:
            try:
                stage_result = await self._run_stage(stage, content)
                self._merge_stage_result(result, stage.name, stage_result)
                result.stages_completed.append(stage.name)
            except Exception as e:
                result.errors.append(f"Stage {stage.name} failed: {str(e)}")

        # Verification loop: detect gaps then fill them
        for round_idx in range(max_verification_rounds):
            try:
                gap_result = await self._detect_gaps(content, result)
                if gap_result is None or gap_result.is_complete or not gap_result.gaps:
                    print(f"Verification round {round_idx + 1}: extraction complete, no gaps found")
                    break
                print(f"Verification round {round_idx + 1}: found {len(gap_result.gaps)} gap(s), filling...")
                await self._fill_gaps(gap_result.gaps, content, result)
            except Exception as e:
                result.errors.append(f"Verification round {round_idx + 1} failed: {str(e)}")
                break

        return result

    async def _detect_gaps(
        self, document_content: str, current_result: FullExtractionResult
    ) -> Optional[GapDetectionSchema]:
        """
        Verify extraction completeness by asking LLM to compare extracted data against the source.

        The LLM receives the full document + current extraction JSON and identifies
        anything that's missing or incorrect.
        """
        import json

        extracted_summary = {
            "program": current_result.program.model_dump() if current_result.program else None,
            "products_count": len(current_result.products),
            "products": [p.model_dump() for p in current_result.products],
            "matrix_cells_count": current_result.cell_count,
            "matrix_cells_sample": [
                c.model_dump() for c in current_result.cells[:10]
            ],
            "rules_count": len(current_result.rules),
            "rules": [r.model_dump() for r in current_result.rules],
            "document_requirements_count": len(current_result.document_requirements),
            "document_requirements": [
                d.model_dump() for d in current_result.document_requirements
            ],
        }

        prompt = f"""You are reviewing the extraction of a mortgage program document.

Below is the ORIGINAL DOCUMENT and the EXTRACTED DATA. Your job is to identify what is missing
or incorrect in the extracted data compared to the document.

Focus on:
1. Matrix cells: Are all FICO/DSCR/LTV combinations captured? Missing rows or columns?
2. Adjustment rules: Cash-Out LTV adjustments, loan amount overlays, property type overlays?
3. State overlays: Any states with restrictions not in the rules list?
4. Borrower restrictions: Foreign National, ITIN, First Time Investor rules missing?
5. Document requirements: Any conditional document requirements not captured?

EXTRACTED DATA:
{json.dumps(extracted_summary, indent=2)}

ORIGINAL DOCUMENT:
{document_content}

Identify ALL gaps. For each gap, provide the stage it belongs to and the exact source text."""

        try:
            conv = self._chak.Conversation(self.llm_uri, api_key=self.api_key)
            result = await conv.asend(prompt, returns=GapDetectionSchema)
            return result
        except Exception as e:
            print(f"Warning: Gap detection failed: {e}")
            return None

    async def _fill_gaps(
        self,
        gaps: List[GapItemSchema],
        document_content: str,
        result: FullExtractionResult,
    ) -> None:
        """
        For each identified gap, run a targeted extraction using only the relevant source text,
        then merge the new data into the existing result.
        """
        # Group gaps by stage for efficiency
        from collections import defaultdict
        gaps_by_stage: Dict[str, List[GapItemSchema]] = defaultdict(list)
        for gap in gaps:
            gaps_by_stage[gap.stage].append(gap)

        for stage_name, stage_gaps in gaps_by_stage.items():
            schema = self._get_stage_schema(stage_name)
            if schema is BaseModel:
                continue  # Unknown stage, skip

            # Combine the source texts from all gaps in this stage
            combined_source = "\n\n---\n\n".join(
                f"Gap: {g.description}\nSource text:\n{g.source_text}"
                for g in stage_gaps
            )

            prompt = (
                f"The following text was identified as missing from a previous extraction pass.\n"
                f"Extract ONLY the data described in each gap from the source text below.\n\n"
                f"{combined_source}"
            )

            try:
                conv = self._chak.Conversation(self.llm_uri, api_key=self.api_key)
                extracted = await conv.asend(prompt, returns=schema)
                if extracted is None:
                    continue
                # Merge into result - append only, never overwrite
                self._merge_stage_result_append(result, stage_name, extracted.model_dump())
            except Exception as e:
                print(f"Warning: Gap fill for stage '{stage_name}' failed: {e}")

    def _merge_stage_result_append(
        self,
        result: FullExtractionResult,
        stage_name: str,
        stage_data: Dict[str, Any],
    ) -> None:
        """
        Append-only merge for gap-fill results.
        Unlike _merge_stage_result, this never overwrites existing data.
        """
        if not stage_data:
            return

        if stage_name == "eligibility_matrix":
            matrices = self._build_matrix(stage_data)
            if matrices:
                result.matrices.extend(matrices)

        elif stage_name in ("borrower_restrictions", "state_overlays"):
            if stage_name == "borrower_restrictions":
                rules = self._build_borrower_rules(stage_data.get("restrictions", []))
            else:
                rules = self._build_state_overlays(stage_data.get("overlays", []))
            result.rules.extend(rules)

        elif stage_name == "document_requirements":
            new_reqs = self._build_doc_requirements(stage_data.get("requirements", []))
            result.document_requirements.extend(new_reqs)
    
    async def _run_stage(self, stage: ExtractionStage, content: str) -> Dict[str, Any]:
        """Run a single extraction stage."""
        
        # Get stage-specific content and schema
        stage_content = self._prepare_stage_content(stage, content)
        schema = self._get_stage_schema(stage.name)
        prompt = self._build_stage_prompt(stage)
        
        full_prompt = f"{prompt}\n\nDocument Content:\n{stage_content}"
        
        # Run LLM extraction - create new Conversation for each stage to avoid state issues
        try:
            conv = self._chak.Conversation(self.llm_uri, api_key=self.api_key)
            extracted = await conv.asend(full_prompt, returns=schema)
            if extracted is None:
                print(f"Warning: Stage {stage.name} returned None")
                return {}
            return extracted.model_dump()
        except Exception as e:
            print(f"Warning: Stage {stage.name} extraction failed: {e}")
            return {}
    
    def _prepare_stage_content(self, stage: ExtractionStage, content: str) -> str:
        """Prepare content for a specific stage."""
        
        # For matrix extraction, try to find and prioritize table content
        if stage.special_handling == "table_extraction":
            # Look for table-like content
            table_content = self._extract_table_sections(content)
            if table_content:
                return table_content[:stage.token_budget * 3]
        
        # Default: truncate to token budget
        return content[:stage.token_budget * 3]
    
    def _extract_table_sections(self, content: str) -> str:
        """Extract table-like sections from content."""
        lines = content.split("\n")
        table_lines = []
        in_table = False
        
        for line in lines:
            # Heuristics for table detection
            is_table_line = (
                "|" in line or
                "---" in line or
                "LTV" in line.upper() or
                "FICO" in line.upper() or
                "DSCR" in line.upper() or
                any(char.isdigit() for char in line) and "%" in line
            )
            
            if is_table_line:
                in_table = True
                table_lines.append(line)
            elif in_table and line.strip() == "":
                in_table = False
                table_lines.append("")
            elif in_table:
                table_lines.append(line)
        
        return "\n".join(table_lines)
    
    def _get_stage_schema(self, stage_name: str) -> Type[BaseModel]:
        """Get the schema for a stage."""
        schemas = {
            "program_overview": ProgramOverviewSchema,
            "product_list": ProductListSchema,
            "eligibility_matrix": EligibilityMatrixSchema,
            "borrower_restrictions": BorrowerRestrictionsSchema,
            "state_overlays": StateOverlaysSchema,
            "document_requirements": DocRequirementsSchema,
        }
        return schemas.get(stage_name, BaseModel)
    
    def _build_stage_prompt(self, stage: ExtractionStage) -> str:
        """Build the prompt for a stage."""
        
        base_prompts = {
            "program_overview": """Extract the basic program information:
- Program name (e.g., "DSCR PRIME", "Non-QM Jumbo")
- Lender name
- Version or effective date
- Whether high-cost loans are allowed
- Maximum points and fees percentage
- Prepayment penalty options (years)
- Prepayment penalty structure

Focus on the document header and overview sections.""",
            
            "product_list": """Extract all mortgage products mentioned in the document:
- Product name (e.g., "5/6 ARM", "30 Year Fixed", "7/6 ARM")
- Product type (Fixed, ARM, SOFR_ARM, Jumbo, etc.)
- For ARM products: index, margin, caps (initial, periodic, lifetime)
- For Fixed products: term in years
- Qualification rate (if specified)

List ALL products found.""",
            
            "eligibility_matrix": """Extract the eligibility matrix/decision table.

This is CRITICAL - extract ALL matrix cells, not just a sample.

For each matrix cell, extract:
- FICO range (min and max)
- DSCR range (min and max)
- Loan amount range (min and max)
- Property type restriction (if any)
- Transaction type (Purchase, Cash-Out, Rate/Term)
- Result LTV percentage
- Result CLTV percentage (if specified)
- Whether eligible

Pay special attention to:
- Table structures with rows and columns
- Range notations like "720+", "700-719", "<=1M"
- Cells that contain LTV percentages
- Different transaction types (Purchase vs Cash-Out)""",
            
            "borrower_restrictions": """Extract borrower-specific restrictions and requirements:

For each borrower type mentioned:
- Foreign National: restrictions, required documents, LTV caps
- ITIN: restrictions, eligibility criteria
- First Time Investor: additional requirements, restrictions
- DACA: eligibility, restrictions
- Non-Permanent Resident: visa requirements, restrictions

Include any special conditions that apply only to specific borrower types.""",
            
            "state_overlays": """Extract state-specific restrictions and overlays:

For each state with special rules:
- State code (CA, TX, FL, IL, NY, etc.)
- Type of restriction (LTV cap, property exclusion, prepay restriction, etc.)
- Restriction value (if applicable)
- Full description of the restriction
- Whether it completely excludes eligibility

Look for sections like "State Restrictions", "State-Specific Requirements", etc.""",
            
            "document_requirements": """Extract document requirements:

For each scenario that triggers additional documents:
- What condition triggers the requirement (loan amount, borrower type, etc.)
- What documents are required
- Description of the requirement

Examples:
- Loan > $2M: second appraisal required
- Foreign National: passport, visa, ACH form required
- ITIN: ITIN card or IRS letter required""",
        }
        
        return base_prompts.get(stage.name, f"Extract: {stage.target}")
    
    def _merge_stage_result(
        self,
        result: FullExtractionResult,
        stage_name: str,
        stage_data: Dict[str, Any],
    ):
        """Merge stage extraction result into final result."""
        
        if not stage_data:
            return
        
        if stage_name == "program_overview":
            result.program = self._build_program(stage_data)
        
        elif stage_name == "product_list":
            result.products = self._build_products(stage_data.get("products", []))
        
        elif stage_name == "eligibility_matrix":
            matrices = self._build_matrix(stage_data)
            if matrices:
                result.matrices.extend(matrices)
        
        elif stage_name == "borrower_restrictions":
            rules = self._build_borrower_rules(stage_data.get("restrictions", []))
            result.rules.extend(rules)
        
        elif stage_name == "state_overlays":
            rules = self._build_state_overlays(stage_data.get("overlays", []))
            result.rules.extend(rules)
        
        elif stage_name == "document_requirements":
            result.document_requirements = self._build_doc_requirements(
                stage_data.get("requirements", [])
            )
    
    def _build_program(self, data: Dict) -> Program:
        """Build Program from stage data."""
        from datetime import datetime
        
        # Get version or use current date
        version = data.get("version")
        if not version:
            version = datetime.now().strftime("%Y-%m")
        
        # Get effective_date or use current date
        effective_date = data.get("effective_date")
        if not effective_date:
            effective_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get allows_high_cost with proper None handling
        allows_high_cost = data.get("allows_high_cost")
        if allows_high_cost is None:
            allows_high_cost = False
        
        return Program(
            name=data.get("program_name", "Unknown"),
            lender=data.get("lender", ""),
            version=version,
            effective_date=effective_date,
            status=ProgramStatus.ACTIVE,
            description=data.get("description"),
            allows_high_cost=allows_high_cost,
            max_points_fees=data.get("max_points_fees"),
            prepay_options=data.get("prepay_options"),
            prepay_structure=data.get("prepay_structure"),
        )
    
    def _build_products(self, products_data: List[Dict]) -> List[Product]:
        """Build Products from stage data."""
        products = []
        for p in products_data:
            arm_params = None
            arm_data = p.get("arm_params")
            if arm_data and isinstance(arm_data, dict) and any(arm_data.values()):
                # Create ARMParams if any field has value
                arm_params = ARMParams(
                    index=arm_data.get("index"),
                    margin=arm_data.get("margin"),
                    initial_cap=arm_data.get("initial_cap"),
                    periodic_cap=arm_data.get("periodic_cap"),
                    lifetime_cap=arm_data.get("lifetime_cap"),
                    adjustment_period=arm_data.get("adjustment_period"),
                )
            
            product = Product(
                name=p.get("name", ""),
                type=self._parse_product_type(p.get("type", "Fixed")),
                description=p.get("description"),
                arm_params=arm_params,
                fixed_term_years=p.get("fixed_term_years"),
                qualify_rate=p.get("qualify_rate"),
            )
            products.append(product)
        
        return products
    
    def _parse_product_type(self, type_str: str) -> ProductType:
        """Parse product type string."""
        type_map = {
            "fixed": ProductType.FIXED,
            "arm": ProductType.ARM,
            "sofr_arm": ProductType.SOFR_ARM,
            "jumbo": ProductType.JUMBO,
            "dscr": ProductType.DSCR,
            "non_qm": ProductType.NON_QM,
            "hybrid": ProductType.HYBRID,
        }
        return type_map.get(type_str.lower().replace(" ", "_"), ProductType.FIXED)
    
    def _build_matrix(self, data: Dict) -> List[Matrix]:
        """Build matrix with embedded cells from stage data."""
        matrices = []
        
        # Handle None or invalid data
        if not data or not isinstance(data, dict):
            return matrices
        
        if data.get("cells"):
            matrix = Matrix(
                name=data.get("matrix_name", "Eligibility Matrix"),
                dimensions=data.get("dimensions", ["FICO", "DSCR", "LoanAmount"]),
            )
            
            for cell_data in data["cells"]:
                if not isinstance(cell_data, dict):
                    continue
                cell = Cell(
                    fico_min=cell_data.get("fico_min"),
                    fico_max=cell_data.get("fico_max"),
                    dscr_min=cell_data.get("dscr_min"),
                    dscr_max=cell_data.get("dscr_max"),
                    loan_min=cell_data.get("loan_min"),
                    loan_max=cell_data.get("loan_max"),
                    result_ltv=cell_data.get("result_ltv"),
                    result_cltv=cell_data.get("result_cltv"),
                    result_eligible=cell_data.get("result_eligible", True),
                )
                matrix.cells.append(cell)  # Add cell to matrix
            
            matrices.append(matrix)
        
        return matrices
    
    def _parse_operator(self, op_str: str) -> OperatorType:
        """Parse operator string."""
        op_map = {
            ">=": OperatorType.GE,
            "<=": OperatorType.LE,
            ">": OperatorType.GT,
            "<": OperatorType.LT,
            "=": OperatorType.EQ,
            "!=": OperatorType.NE,
            "in": OperatorType.IN,
            "between": OperatorType.BETWEEN,
        }
        return op_map.get(op_str, OperatorType.GE)

    def _build_borrower_rules(self, restrictions_data: List[Dict]) -> List[Rule]:
        """Build Rules from borrower restrictions."""
        rules = []
        for r in restrictions_data:
            borrower_type = r.get('borrower_type', 'Unknown')
            restrictions = r.get("restrictions", [])
            action_text = "; ".join(restrictions) if restrictions else "No specific restrictions"
            
            rule = Rule(
                name=f"{borrower_type} Restrictions",
                category="borrower",
                trigger=f"Borrower type is {borrower_type}",
                action=action_text,
            )
            rules.append(rule)
        
        return rules
    
    def _build_state_overlays(self, overlays_data: List[Dict]) -> List[Rule]:
        """Build Rules from state overlay data."""
        rules = []
        for o in overlays_data:
            state = o.get("state", "Unknown")
            desc = o.get("restriction_description", "")
            
            rule = Rule(
                name=f"State Rule - {state}",
                category="state",
                trigger=f"Loan in state {state}",
                action=desc,
                excludes=o.get("excludes", False),
            )
            rules.append(rule)
        
        return rules
    
    def _build_doc_requirements(self, reqs_data: List[Dict]) -> List[DocumentRequirement]:
        """Build DocumentRequirements from stage data."""
        requirements = []
        for r in reqs_data:
            req = DocumentRequirement(
                trigger_field=r.get("trigger_field", ""),
                trigger_operator=self._parse_operator(r.get("trigger_operator", ">")),
                trigger_value=r.get("trigger_value"),
                required_documents=r.get("required_documents", []),
                description=r.get("description"),
            )
            requirements.append(req)
        
        return requirements
