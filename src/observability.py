"""
LangSmith observability helpers.

All LangGraph runs automatically appear in the LangSmith dashboard when
LANGCHAIN_TRACING_V2=true is set. This module adds structured metadata
to each run so traces are filterable by query, model, threshold, and
routing decision.

Usage:
    config = make_run_config(query, threshold=0.6, source="api")
    result = graph.invoke(initial_state, config=config)
"""

import os

from langchain_core.runnables import RunnableConfig


def langsmith_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def make_run_config(
    query: str,
    threshold: float | None = None,
    source: str = "api",
    experiment_name: str | None = None,
    **extra_metadata,
) -> RunnableConfig:
    """
    Build a RunnableConfig that attaches metadata to a LangSmith trace.

    Each field appears as a searchable/filterable attribute in the
    LangSmith UI under Metadata → Run.
    """
    from src.config import settings

    metadata: dict = {
        "query_preview": query[:120],
        "relevance_threshold": threshold or settings.relevance_threshold,
        "llm_model": settings.llm_model,
        "eval_model": settings.eval_model,
        "top_k": settings.top_k,
        "source": source,
        **extra_metadata,
    }

    tags = ["crag", source]
    if experiment_name:
        tags.append(experiment_name)

    return RunnableConfig(
        run_name="adaptive-rag",
        tags=tags,
        metadata=metadata,
    )


def log_outcome(config: RunnableConfig, state: dict) -> None:
    """
    Patch the run config metadata with post-execution facts.

    LangSmith allows updating metadata after a run completes via the
    langsmith client; here we add web_search outcome and avg_relevance
    so these are visible as filters in the dashboard.
    """
    if not langsmith_enabled():
        return

    try:
        from langsmith import Client

        client = Client()
        run_id = config.get("run_id")
        if run_id:
            client.update_run(
                run_id,
                extra={
                    "web_search_triggered": state.get("web_search_triggered"),
                    "avg_relevance": round(state.get("avg_relevance", 0.0), 3),
                    "sources_count": len(state.get("sources", [])),
                },
            )
    except Exception:
        pass  # never crash the pipeline due to observability failure
