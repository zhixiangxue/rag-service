"""Demo query API with LLM analysis (temporary, may be removed in future)."""
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..schemas import ApiResponse
from .. import config
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers.basic import VectorRetriever
from zag.postprocessors import Reranker, LLMSelector

# TODO: This is a temporary demo endpoint for the web UI
# Will be removed or refactored when proper query pipeline is implemented

router = APIRouter(prefix="/query", tags=["demo"])

# Constants
TOP_K = 20
FINAL_TOP_K = 5
RERANKER_MODEL = "cohere/rerank-english-v3.0"
LLM_SELECTOR_URI = "openai/gpt-4o-mini"
LLM_URI = "openai/gpt-4o"

# Global components cache (lazy initialized)
_demo_components = {}


def _get_retriever():
    """Get or create retriever (cached)."""
    if "retriever" not in _demo_components:
        # TODO: Use dataset_id instead of hardcoded collection
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name="mortgage_guidelines",  # TODO: Remove hardcode
            embedder=embedder,
            timeout=180  # 3 minutes for slow LLM pipeline
        )
        _demo_components["retriever"] = VectorRetriever(vector_store=vector_store, top_k=TOP_K)
    return _demo_components["retriever"]


class WebQueryRequest(BaseModel):
    """Web query request model."""
    query: str = Field(..., description="User's question")
    top_k: Optional[int] = Field(FINAL_TOP_K, description="Number of results to return")


class RelevanceAnalysis(BaseModel):
    """Relevance analysis result."""
    is_relevant: bool
    confidence: str
    reason: str
    relevant_excerpts: List[str]


class WebQueryResult(BaseModel):
    """Single web query result."""
    unit_id: str
    score: float
    content: str
    metadata: Optional[dict] = None
    analysis: Optional[RelevanceAnalysis] = None


class WebQueryResponse(BaseModel):
    """Web query response model."""
    query: str
    results: List[WebQueryResult]
    timing: dict
    total_time: float


async def analyze_relevance(query: str, content: str) -> RelevanceAnalysis:
    """Analyze relevance between query and content using LLM."""
    try:
        import chak
        
        conv = chak.Conversation(LLM_URI, api_key=config.OPENAI_API_KEY)
        
        analysis_prompt = f"""Analyze the relevance between the query and retrieved content.

User Query: {query}

Retrieved Content:
{content}

Please analyze:
1. Is this content relevant to the query?
2. What is the relevance level (high/medium/low)?
3. Why is it relevant or not relevant?
4. If relevant, extract the relevant excerpts verbatim (can be multiple segments)

Note: When extracting excerpts, copy the original text exactly without any modifications.
"""
        
        analysis = await conv.asend(analysis_prompt, returns=RelevanceAnalysis)
        return analysis
        
    except Exception as e:
        return RelevanceAnalysis(
            is_relevant=True,
            confidence="unknown",
            reason=f"Analysis failed: {str(e)}",
            relevant_excerpts=[]
        )


@router.post("/web", response_model=WebQueryResponse)
async def query_web(request: WebQueryRequest):
    """
    Process a query with full postprocessing pipeline (for web UI demo).
    
    This endpoint includes:
    - Vector retrieval
    - Reranking (Cohere)
    - LLM Selector (GPT-4o-mini)
    - Relevance analysis (GPT-4o)
    
    WARNING: This is slow (~20s) due to multiple LLM calls.
    For production use, call /datasets/{dataset_id}/query/vector instead.
    
    TODO: This endpoint uses hardcoded collection name 'mortgage_guidelines'.
    Will be refactored to use dataset_id when demo is replaced with proper UI.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    timing = {}
    
    try:
        # Stage 1: Retrieval
        start_time = time.time()
        try:
            retriever = _get_retriever()
            results = retriever.retrieve(request.query)
            timing["retrieval"] = time.time() - start_time
        except Exception as e:
            print(f"Error: Retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval failed: {str(e)}"
            )
        
        if not results:
            return WebQueryResponse(
                query=request.query,
                results=[],
                timing=timing,
                total_time=timing["retrieval"]
            )
        
        # Stage 2: Reranking
        start_time = time.time()
        try:
            # TODO: Get API key from config
            cohere_key = config.OPENAI_API_KEY  # TODO: Should be COHERE_API_KEY
            reranker = Reranker(RERANKER_MODEL, api_key=cohere_key)
            results_reranked = reranker.rerank(
                request.query,
                results[:TOP_K],
                top_k=None
            )
            timing["reranking"] = time.time() - start_time
        except Exception as e:
            print(f"Warning: Reranking failed, using original results: {e}")
            results_reranked = results[:TOP_K]
            timing["reranking"] = time.time() - start_time
        
        # Stage 3: LLM Selector
        start_time = time.time()
        try:
            selector = LLMSelector(llm_uri=LLM_SELECTOR_URI, api_key=config.OPENAI_API_KEY)
            results = await selector.aprocess(request.query, results_reranked)
        except Exception as e:
            print(f"Warning: LLM Selector failed, using reranked results: {e}")
            results = results_reranked
        timing["selector"] = time.time() - start_time
        
        # Limit to requested top k
        results = results[:request.top_k]
        
        # Stage 4: Analyze relevance
        analyzed_results = []
        start_time = time.time()
        
        for result in results:
            try:
                analysis = await analyze_relevance(request.query, result.content)
            except Exception as e:
                print(f"Warning: Relevance analysis failed: {e}")
                analysis = RelevanceAnalysis(
                    is_relevant=True,
                    confidence="unknown",
                    reason="Analysis failed",
                    relevant_excerpts=[]
                )
            
            # Prepare metadata (exclude 'document' field)
            metadata = None
            try:
                if hasattr(result, 'metadata') and result.metadata:
                    if isinstance(result.metadata, dict):
                        metadata = {k: v for k, v in result.metadata.items() if k != 'document'}
                    else:
                        metadata = {
                            k: v for k, v in result.metadata.__dict__.items()
                            if not k.startswith('_') and k != 'document'
                        }
            except Exception as e:
                print(f"Warning: Metadata extraction failed: {e}")
                metadata = {}
            
            analyzed_results.append(WebQueryResult(
                unit_id=result.unit_id,
                score=result.score,
                content=result.content,
                metadata=metadata,
                analysis=analysis
            ))
        
        timing["analysis"] = time.time() - start_time
        total_time = sum(timing.values())
        
        return WebQueryResponse(
            query=request.query,
            results=analyzed_results,
            timing=timing,
            total_time=total_time
        )
        
    except Exception as e:
        print(f"Error: Unexpected error in query_web: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
