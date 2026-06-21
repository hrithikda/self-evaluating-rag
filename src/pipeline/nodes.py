"""
LangGraph node implementations for the Adaptive RAG pipeline.

Pipeline flow:
  retrieve → rerank → evaluate_relevance → [web_search →] filter_context → generate
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.ingestion.vectorstore import get_vectorstore
from src.pipeline.state import RAGState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-encoder re-ranker (loaded once, lives in process memory)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder reranker (first call only)…")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM outputs
# ---------------------------------------------------------------------------


class DocumentRelevance(BaseModel):
    """Relevance judgment for a single retrieved document."""

    score: float = Field(
        ge=0.0, le=1.0,
        description="0.0 = completely irrelevant, 1.0 = directly answers the query",
    )
    reason: str = Field(description="One-sentence explanation of the score")


class FilteredKnowledge(BaseModel):
    """Result of decompose-and-recompose filtering on a document."""

    relevant_strips: list[str] = Field(
        description="Verbatim sentences/phrases that are directly relevant to the query"
    )
    removed_count: int = Field(
        description="Number of sentences discarded as irrelevant"
    )


# ---------------------------------------------------------------------------
# LLM factory helpers
# ---------------------------------------------------------------------------


def _llm(model: str | None = None, temperature: float | None = None) -> ChatAnthropic:
    return ChatAnthropic(
        model=model or settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )


# ---------------------------------------------------------------------------
# Node 1: Retrieve from ChromaDB
# ---------------------------------------------------------------------------


def retrieve_node(state: RAGState) -> dict:
    query = state["query"]
    vs = get_vectorstore()

    try:
        results = vs.similarity_search_with_relevance_scores(query, k=settings.top_k)
    except Exception as exc:
        logger.warning("ChromaDB retrieval failed: %s", exc)
        results = []

    documents = [
        {
            "content": doc.page_content,
            "metadata": {**doc.metadata, "source_type": "local"},
            "source": "local",
            "relevance_score": float(score),
        }
        for doc, score in results
    ]

    return {
        "documents": documents,
        "web_documents": [],
        "web_search_triggered": False,
        "pipeline_log": [
            f"[RETRIEVE] ChromaDB → {len(documents)} documents fetched (top-{settings.top_k})"
        ],
    }


# ---------------------------------------------------------------------------
# Node 2: Cross-encoder re-ranking (local, no API call)
# ---------------------------------------------------------------------------


def rerank_node(state: RAGState) -> dict:
    """
    Re-score documents with a cross-encoder that sees the (query, doc) pair —
    unlike embedding similarity which compares independently encoded vectors.
    Cross-encoders capture fine-grained semantic interactions and produce
    significantly better relevance signals, especially for short queries.
    """
    query = state["query"]
    documents = state["documents"]

    if not documents:
        return {
            "documents": [],
            "pipeline_log": ["[RERANK] No documents to re-rank"],
        }

    try:
        reranker = _get_reranker()
        pairs = [(query, doc["content"][:512]) for doc in documents]
        raw_scores: np.ndarray = reranker.predict(pairs)

        # Sigmoid-normalize raw logits to [0, 1] for interpretable display
        normalized = (1 / (1 + np.exp(-raw_scores))).tolist()

        # Attach rerank score and sort documents best-first
        for doc, score in zip(documents, normalized):
            doc["rerank_score"] = round(float(score), 4)

        documents = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        top_scores = [f"{d['rerank_score']:.2f}" for d in documents]

        return {
            "documents": documents,
            "pipeline_log": [
                f"[RERANK] Cross-encoder scores (best-first): {top_scores}"
            ],
        }

    except Exception as exc:
        logger.warning("Reranker failed (%s) — using original retrieval order", exc)
        return {
            "documents": documents,
            "pipeline_log": [f"[RERANK] Skipped (model unavailable): {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 3: Evaluate relevance with LLM
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _score_document(evaluator, query: str, doc: dict) -> DocumentRelevance:
    prompt = (
        f"Query: {query}\n\n"
        f"Document excerpt:\n{doc['content'][:1200]}\n\n"
        "Rate how relevant this document is to answering the query.\n"
        "0.0 = completely off-topic\n"
        "0.5 = tangentially related\n"
        "1.0 = directly answers the query"
    )
    return evaluator.invoke(prompt)


def evaluate_relevance_node(state: RAGState) -> dict:
    query = state["query"]
    documents = state["documents"]

    if not documents:
        return {
            "relevance_scores": [],
            "avg_relevance": 0.0,
            "pipeline_log": ["[EVALUATE] No documents to score → avg=0.00, triggering web search"],
        }

    # Use lighter/faster model for evaluation to keep costs down
    evaluator = _llm(model=settings.eval_model).with_structured_output(DocumentRelevance)

    # Score documents in parallel — each call is independent
    scores: list[float] = [0.0] * len(documents)
    with ThreadPoolExecutor(max_workers=min(5, len(documents))) as pool:
        future_to_idx = {
            pool.submit(_score_document, evaluator, query, doc): i
            for i, doc in enumerate(documents)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result: DocumentRelevance = future.result()
                scores[idx] = result.score
                documents[idx]["relevance_score"] = result.score
                documents[idx]["relevance_reason"] = result.reason
            except Exception as exc:
                logger.warning("Scoring failed for doc %d: %s", idx, exc)
                scores[idx] = 0.5
                documents[idx]["relevance_score"] = 0.5
                documents[idx]["relevance_reason"] = "Evaluation unavailable"

    avg = sum(scores) / len(scores)
    verdict = "✓ sufficient" if avg >= settings.relevance_threshold else "✗ below threshold → web fallback"

    return {
        "documents": documents,
        "relevance_scores": scores,
        "avg_relevance": avg,
        "pipeline_log": [
            f"[EVALUATE] Scores: {[f'{s:.2f}' for s in scores]} | "
            f"avg={avg:.2f} (threshold={settings.relevance_threshold}) {verdict}"
        ],
    }


# ---------------------------------------------------------------------------
# Conditional router (not a node — called by add_conditional_edges)
# ---------------------------------------------------------------------------


def route_after_evaluation(state: RAGState) -> str:
    if state["avg_relevance"] < settings.relevance_threshold or not state["documents"]:
        return "web_search"
    return "filter_context"


# ---------------------------------------------------------------------------
# Node 3: Web search fallback (Tavily)
# ---------------------------------------------------------------------------


def web_search_node(state: RAGState) -> dict:
    query = state["query"]

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=5,
            include_raw_content=True,
        )
        raw_results = response.get("results", [])
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        raw_results = []

    web_docs = []
    for r in raw_results:
        content = r.get("raw_content") or r.get("content", "")
        if content:
            web_docs.append(
                {
                    "content": content,
                    "metadata": {
                        "url": r.get("url", ""),
                        "title": r.get("title", "Unknown"),
                        "source_type": "web",
                    },
                    "source": "web",
                    "relevance_score": None,
                }
            )

    return {
        "web_documents": web_docs,
        "web_search_triggered": True,
        "pipeline_log": [
            f"[WEB SEARCH] Triggered — Tavily returned {len(web_docs)} results "
            f"(local avg score was {state['avg_relevance']:.2f})"
        ],
    }


# ---------------------------------------------------------------------------
# Node 4: Decompose-and-recompose context filter
# ---------------------------------------------------------------------------


def _filter_document(filter_llm, query: str, doc: dict) -> list[str]:
    content = doc["content"]
    sentences = [
        s.strip()
        for s in content.replace("\n", " ").split(". ")
        if len(s.strip()) > 25
    ]
    if not sentences:
        return []

    prompt = (
        f"Query: {query}\n\n"
        "Below are sentences from a retrieved document. "
        "Select ONLY the sentences that directly help answer the query. "
        "Return them verbatim.\n\n"
        + "\n".join(f"- {s}" for s in sentences[:60])
    )

    try:
        result: FilteredKnowledge = filter_llm.invoke(prompt)
        return result.relevant_strips
    except Exception as exc:
        logger.warning("Filter LLM failed: %s — using first 3 sentences", exc)
        return sentences[:3]


def filter_context_node(state: RAGState) -> dict:
    query = state["query"]
    all_docs = state["documents"] + state.get("web_documents", [])

    if not all_docs:
        return {
            "filtered_context": "No context available.",
            "pipeline_log": ["[FILTER] No documents — context is empty"],
        }

    filter_llm = _llm(model=settings.eval_model).with_structured_output(FilteredKnowledge)

    # Filter each document in parallel
    all_strips: list[str] = []
    with ThreadPoolExecutor(max_workers=min(5, len(all_docs))) as pool:
        futures = [pool.submit(_filter_document, filter_llm, query, doc) for doc in all_docs]
        for future in as_completed(futures):
            try:
                all_strips.extend(future.result())
            except Exception as exc:
                logger.warning("Filter future failed: %s", exc)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_strips: list[str] = []
    for strip in all_strips:
        key = strip.lower().strip()
        if key not in seen and strip:
            seen.add(key)
            unique_strips.append(strip)

    context = " ".join(unique_strips) if unique_strips else "\n\n".join(
        doc["content"][:600] for doc in all_docs[:3]
    )

    total_docs = len(all_docs)
    source_breakdown = (
        f"{len(state['documents'])} local + {len(state.get('web_documents', []))} web"
    )

    return {
        "filtered_context": context,
        "pipeline_log": [
            f"[FILTER] Recomposed {len(unique_strips)} strips from {total_docs} docs ({source_breakdown})"
        ],
    }


# ---------------------------------------------------------------------------
# Node 5: Generate final answer
# ---------------------------------------------------------------------------


def generate_node(state: RAGState) -> dict:
    query = state["query"]
    context = state["filtered_context"]
    chat_history = state.get("chat_history") or []

    # Build structured source list for attribution
    sources: list[dict] = []
    for doc in state["documents"]:
        sources.append(
            {
                "content_preview": doc["content"][:300],
                "source": "local",
                "score": doc.get("relevance_score"),
                "rerank_score": doc.get("rerank_score"),
                "reason": doc.get("relevance_reason", ""),
                "metadata": doc.get("metadata", {}),
            }
        )
    for doc in state.get("web_documents", []):
        sources.append(
            {
                "content_preview": doc["content"][:300],
                "source": "web",
                "url": doc.get("metadata", {}).get("url", ""),
                "title": doc.get("metadata", {}).get("title", ""),
                "metadata": doc.get("metadata", {}),
            }
        )

    gen_llm = _llm(temperature=0.1)

    # Build conversation history block for multi-turn context
    history_block = ""
    if chat_history:
        lines = []
        for msg in chat_history[-6:]:  # keep last 3 turns (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    web_note = (
        "Note: Local knowledge base had insufficient context — web search results "
        "were incorporated.\n\n"
        if state.get("web_search_triggered")
        else ""
    )

    prompt = (
        f"You are a precise, expert AI assistant.\n\n"
        f"{web_note}"
        f"{history_block}"
        f"Answer the following question using ONLY the provided context. "
        f"Be comprehensive but concise. "
        f"If context is insufficient to fully answer, say so clearly.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context[:5000]}\n\n"
        f"Answer:"
    )

    try:
        response = gen_llm.invoke(prompt)
        answer = response.content
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        answer = f"Generation failed: {exc}"

    history_note = f" | {len(chat_history)} prior turns in context" if chat_history else ""

    return {
        "answer": answer,
        "sources": sources,
        "pipeline_log": [
            f"[GENERATE] Answer produced using "
            f"{'local + web' if state.get('web_search_triggered') else 'local'} context"
            f"{history_note}"
        ],
    }
