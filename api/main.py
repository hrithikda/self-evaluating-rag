"""
FastAPI application exposing the Adaptive RAG pipeline.

Endpoints
---------
GET  /health          — liveness + collection stats
POST /query           — run full CRAG pipeline
POST /ingest          — add documents to ChromaDB
GET  /collection/info — vector store metadata
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDoc,
)
from src.config import settings
from src.ingestion.vectorstore import add_documents, collection_size, get_vectorstore
from src.pipeline.graph import build_graph, make_initial_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Pre-build graph once at startup
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Warming up vector store and LangGraph…")
    get_vectorstore()        # loads ChromaDB + embeddings
    _graph = build_graph()   # compiles LangGraph DAG
    logger.info("Ready. Collection size: %d", collection_size())
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Adaptive RAG API",
    description=(
        "Corrective RAG pipeline: ChromaDB retrieval → LLM relevance scoring → "
        "conditional Tavily web fallback → decompose-recompose filtering → Claude generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        collection_size=collection_size(),
        model=settings.llm_model,
        threshold=settings.relevance_threshold,
    )


@app.get("/collection/info", tags=["meta"])
def collection_info():
    vs = get_vectorstore()
    col = vs._collection
    return {"name": col.name, "count": col.count()}


@app.post("/query", response_model=QueryResponse, tags=["pipeline"])
def query(request: QueryRequest):
    # Allow per-request threshold/top_k overrides
    original_threshold = settings.relevance_threshold
    original_top_k = settings.top_k

    if request.relevance_threshold is not None:
        settings.relevance_threshold = request.relevance_threshold
    if request.top_k is not None:
        settings.top_k = request.top_k

    try:
        t0 = time.perf_counter()
        state = _graph.invoke(make_initial_state(request.query))
        elapsed = time.perf_counter() - t0
        logger.info("Query processed in %.2fs (web=%s)", elapsed, state["web_search_triggered"])
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        settings.relevance_threshold = original_threshold
        settings.top_k = original_top_k

    sources = [SourceDoc(**s) for s in state["sources"]]

    return QueryResponse(
        query=request.query,
        answer=state["answer"],
        web_search_triggered=state["web_search_triggered"],
        avg_relevance_score=state["avg_relevance"],
        sources=sources,
        pipeline_log=state["pipeline_log"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["data"])
def ingest(request: IngestRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    try:
        ids = add_documents(request.texts, request.metadatas)
    except Exception as exc:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(added=len(ids), collection_size=collection_size())
