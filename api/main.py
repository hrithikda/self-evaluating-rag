"""
FastAPI application exposing the Adaptive RAG pipeline.

Endpoints
---------
GET  /health          — liveness + LangSmith status
POST /query           — full CRAG pipeline with optional chat history
POST /ingest          — add documents to ChromaDB
GET  /collection/info — vector store metadata
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ChatMessage,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDoc,
)
from src.config import settings
from src.ingestion.vectorstore import add_documents, collection_size, get_vectorstore
from src.observability import langsmith_enabled, log_outcome, make_run_config
from src.pipeline.graph import build_graph, make_initial_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Warming up vector store and LangGraph…")
    get_vectorstore()
    _graph = build_graph()
    logger.info(
        "Ready. Collection size: %d | LangSmith: %s",
        collection_size(),
        "enabled" if langsmith_enabled() else "disabled",
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Adaptive RAG API",
    description=(
        "Corrective RAG: ChromaDB → cross-encoder rerank → LLM relevance scoring → "
        "conditional Tavily web fallback → decompose-recompose → Claude generation. "
        "Supports multi-turn conversation via chat_history."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        collection_size=collection_size(),
        model=settings.llm_model,
        threshold=settings.relevance_threshold,
        langsmith_enabled=langsmith_enabled(),
    )


@app.get("/collection/info", tags=["meta"])
def collection_info():
    vs = get_vectorstore()
    col = vs._collection
    return {"name": col.name, "count": col.count()}


@app.post("/query", response_model=QueryResponse, tags=["pipeline"])
def query(request: QueryRequest):
    original_threshold = settings.relevance_threshold
    original_top_k = settings.top_k

    if request.relevance_threshold is not None:
        settings.relevance_threshold = request.relevance_threshold
    if request.top_k is not None:
        settings.top_k = request.top_k

    history = [ChatMessage(role=m.role, content=m.content) for m in request.chat_history]
    run_config = make_run_config(
        request.query,
        threshold=settings.relevance_threshold,
        source="api",
        history_turns=len(history),
    )

    try:
        t0 = time.perf_counter()
        initial = make_initial_state(
            request.query,
            chat_history=[{"role": m.role, "content": m.content} for m in history],
        )
        state = _graph.invoke(initial, config=run_config)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Query in %.2fs | web=%s | avg_rel=%.2f",
            elapsed,
            state["web_search_triggered"],
            state["avg_relevance"],
        )
        log_outcome(run_config, state)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        settings.relevance_threshold = original_threshold
        settings.top_k = original_top_k

    # Return updated history so stateless clients can carry it forward
    updated_history = list(history) + [
        ChatMessage(role="user", content=request.query),
        ChatMessage(role="assistant", content=state["answer"]),
    ]

    return QueryResponse(
        query=request.query,
        answer=state["answer"],
        web_search_triggered=state["web_search_triggered"],
        avg_relevance_score=state["avg_relevance"],
        sources=[SourceDoc(**s) for s in state["sources"]],
        pipeline_log=state["pipeline_log"],
        updated_history=updated_history,
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
