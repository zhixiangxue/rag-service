#!/usr/bin/env python3
"""
FastAPI Server for RAG Query Service
Provides RESTful API endpoints for document retrieval and question answering
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import chak
from zag.postprocessors import Reranker, LLMSelector
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers import VectorRetriever

# Load environment variables - auto-search from current dir up to project root
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is required. Please set it in .env file or environment variable."
    )
if not COHERE_API_KEY:
    raise ValueError(
        "COHERE_API_KEY is required. Please set it in .env file or environment variable."
    )

QDRANT_HOST = os.getenv("QDRANT_HOST", "13.56.109.233")
# QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "16333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "16334"))
COLLECTION_NAME = "mortgage_guidelines"
EMBEDDING_URI = "openai/text-embedding-3-small"
RERANKER_MODEL = "cohere/rerank-english-v3.0"
TOP_K = 20
FINAL_TOP_K = 5
LLM_SELECTOR_URI = "openai/gpt-4o-mini"
LLM_URI = "openai/gpt-4o"

# Global components
components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup"""
    print("Initializing RAG components...")
    
    components["reranker"] = Reranker(RERANKER_MODEL, api_key=COHERE_API_KEY)
    components["selector"] = LLMSelector(llm_uri=LLM_SELECTOR_URI, api_key=OPENAI_API_KEY)
    components["embedder"] = Embedder(EMBEDDING_URI, api_key=OPENAI_API_KEY)
    components["vector_store"] = QdrantVectorStore.server(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=False,  # Disable gRPC to avoid timeout issues
        collection_name=COLLECTION_NAME,
        embedder=components["embedder"]
    )
    components["retriever"] = VectorRetriever(
        vector_store=components["vector_store"],
        top_k=TOP_K
    )
    
    print("✓ RAG components initialized successfully")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RAG Query API",
    description="RESTful API for document retrieval and question answering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="User's question")
    top_k: Optional[int] = Field(FINAL_TOP_K, description="Number of results to return")


class RelevanceAnalysis(BaseModel):
    """Relevance analysis result"""
    is_relevant: bool
    confidence: str
    reason: str
    relevant_excerpts: List[str]


class QueryResult(BaseModel):
    """Single query result"""
    unit_id: str
    score: float
    content: str
    metadata: Optional[dict] = None
    analysis: Optional[RelevanceAnalysis] = None


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    results: List[QueryResult]
    timing: dict
    total_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: dict


# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend page"""
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "retriever": "ready" if "retriever" in components else "not_ready",
            "reranker": "ready" if "reranker" in components else "not_ready",
            "selector": "ready" if "selector" in components else "not_ready",
        }
    }


async def analyze_relevance(query: str, content: str) -> RelevanceAnalysis:
    """Analyze relevance between query and content"""
    try:
        conv = chak.Conversation(LLM_URI, api_key=OPENAI_API_KEY)
        
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
            reason=f"分析失败: {str(e)}",
            relevant_excerpts=[]
        )


@app.post("/query/web", response_model=QueryResponse)
async def query_web(request: QueryRequest):
    """
    Process a query and return relevant results (with full postprocessing for web UI)
    
    Args:
        request: Query request containing the user's question
        
    Returns:
        Query response with results and timing information
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    timing = {}
    
    try:
        # Stage 1: Retrieval (critical - must succeed)
        start_time = time.time()
        try:
            results = components["retriever"].retrieve(request.query)
            timing["retrieval"] = time.time() - start_time
        except Exception as e:
            print(f"Error: Retrieval failed: {e}")
            return QueryResponse(
                query=request.query,
                results=[],
                timing={"retrieval": time.time() - start_time},
                total_time=time.time() - start_time
            )
        
        if not results:
            return QueryResponse(
                query=request.query,
                results=[],
                timing=timing,
                total_time=timing["retrieval"]
            )
        
        # Stage 2: Reranking (optional - fall back to original results)
        start_time = time.time()
        try:
            results_reranked = components["reranker"].rerank(
                request.query,
                results[:TOP_K],
                top_k=None
            )
            timing["reranking"] = time.time() - start_time
        except Exception as e:
            print(f"Warning: Reranking failed, using original results: {e}")
            results_reranked = results[:TOP_K]
            timing["reranking"] = time.time() - start_time
        
        # Stage 3: LLM Selector (optional - fall back to reranked results)
        start_time = time.time()
        try:
            results = await components["selector"].aprocess(request.query, results_reranked)
        except Exception as e:
            print(f"Warning: LLM Selector failed, using reranked results: {e}")
            results = results_reranked
        timing["selector"] = time.time() - start_time
        
        # Limit to requested top k
        results = results[:request.top_k]
        
        # Stage 4: Analyze relevance for each result (optional)
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
                    reason="质量分析失败",
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
            
            analyzed_results.append(QueryResult(
                unit_id=result.unit_id,
                score=result.score,
                content=result.content,
                metadata=metadata,
                analysis=analysis
            ))
        
        timing["analysis"] = time.time() - start_time
        total_time = sum(timing.values())
        
        return QueryResponse(
            query=request.query,
            results=analyzed_results,
            timing=timing,
            total_time=total_time
        )
        
    except Exception as e:
        print(f"Error: Unexpected error in query_web: {e}")
        # Return empty results instead of raising exception
        return QueryResponse(
            query=request.query,
            results=[],
            timing=timing if timing else {},
            total_time=sum(timing.values()) if timing else 0
        )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query with basic postprocessing (for developers)
    
    This endpoint skips LLM-based postprocessors for faster responses:
    - No LLM Selector
    - No relevance analysis
    
    Args:
        request: Query request containing the user's question
        
    Returns:
        Query response with results and timing information
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    timing = {}
    
    try:
        # Stage 1: Retrieval (critical - must succeed)
        start_time = time.time()
        try:
            results = components["retriever"].retrieve(request.query)
            timing["retrieval"] = time.time() - start_time
        except Exception as e:
            print(f"Error: Retrieval failed: {e}")
            return QueryResponse(
                query=request.query,
                results=[],
                timing={"retrieval": time.time() - start_time},
                total_time=time.time() - start_time
            )
        
        if not results:
            return QueryResponse(
                query=request.query,
                results=[],
                timing=timing,
                total_time=timing["retrieval"]
            )
        
        # Stage 2: Reranking only (optional - fall back to original results)
        start_time = time.time()
        try:
            results_reranked = components["reranker"].rerank(
                request.query,
                results[:TOP_K],
                top_k=None
            )
            timing["reranking"] = time.time() - start_time
        except Exception as e:
            print(f"Warning: Reranking failed, using original results: {e}")
            results_reranked = results[:TOP_K]
            timing["reranking"] = time.time() - start_time
        
        # Limit to requested top k
        results = results_reranked[:request.top_k]
        
        # Prepare results without analysis
        simple_results = []
        for result in results:
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
            
            simple_results.append(QueryResult(
                unit_id=result.unit_id,
                score=result.score,
                content=result.content,
                metadata=metadata,
                analysis=None  # No analysis in simple mode
            ))
        
        total_time = sum(timing.values())
        
        return QueryResponse(
            query=request.query,
            results=simple_results,
            timing=timing,
            total_time=total_time
        )
        
    except Exception as e:
        print(f"Error: Unexpected error in query: {e}")
        # Return empty results instead of raising exception
        return QueryResponse(
            query=request.query,
            results=[],
            timing=timing if timing else {},
            total_time=sum(timing.values()) if timing else 0
        )


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("Starting RAG Query Server")
    print("=" * 60)
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Web Interface: http://localhost:8000")
    print(f"Query API (fast): http://localhost:8000/query")
    print(f"Query API (with LLM analysis): http://localhost:8000/query/web")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
