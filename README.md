# Adaptive RAG — Corrective Retrieval-Augmented Generation

A production-grade RAG pipeline that **evaluates the quality of its own retrieved context** before answering — and automatically falls back to live web search when local confidence is low.

Built with LangGraph, ChromaDB, and the Anthropic Claude API.

---

## The Problem with Standard RAG

Standard RAG pipelines blindly pass whatever they retrieve into the LLM. When the knowledge base is stale, incomplete, or the query is out-of-distribution, the model either hallucinates confidently or gives non-answers.

This project solves that with a **self-correcting retrieval loop**:

```
User Query
    │
    ▼
ChromaDB Retrieval (top-k)
    │
    ▼
LLM Relevance Evaluation ◄── scores each document 0.0–1.0 in parallel
    │
    ├── avg ≥ threshold (0.6) ──► Decompose & Recompose ──► Generate
    │
    └── avg < threshold ────────► Tavily Web Search
                                        │
                                        ▼
                                 Decompose & Recompose ──► Generate
```

The **decompose-and-recompose** step (from the CRAG paper) strips noise from mixed-source context by filtering to only the sentences directly relevant to the query before generation.

---

## Architecture

| Layer | Component | Role |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Stateful, conditional pipeline DAG |
| Generation | Claude (`claude-sonnet-4-6`) | Final answer synthesis |
| Evaluation | Claude (`claude-haiku-4-5`) | Relevance scoring (fast, cheap) |
| Filtering | Claude (`claude-haiku-4-5`) | Decompose-and-recompose (parallel) |
| Vector Store | ChromaDB + HNSW | Local document retrieval |
| Embeddings | `all-MiniLM-L6-v2` | Local, no API key needed |
| Web Search | Tavily API | Live fallback retrieval |
| API | FastAPI | Production-ready REST endpoints |
| UI | Streamlit + Plotly | Interactive demo with live pipeline visualization |

### LangGraph Nodes

| # | Node | Description |
|---|---|---|
| 1 | `retrieve` | Embeds query, runs ChromaDB similarity search |
| 2 | `evaluate_relevance` | Scores each doc in parallel with LLM judge |
| — | `route` | Conditional edge: local vs web based on avg score |
| 3 | `web_search` | Tavily API call (only triggered when avg < threshold) |
| 4 | `filter_context` | Decompose-and-recompose noise reduction across all sources |
| 5 | `generate` | Final answer with source attribution |

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo>
cd self-evaluating-rag
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and TAVILY_API_KEY
```

Get API keys:
- Anthropic: https://console.anthropic.com
- Tavily: https://tavily.com (free tier: 1000 searches/month)

### 3. Ingest sample documents

```bash
make ingest
```

This loads 5 AI/ML knowledge base files from `data/sample_docs/` into ChromaDB, chunked and embedded locally.

### 4. Run the Streamlit demo

```bash
make app
# → http://localhost:8501
```

### 5. Run the REST API

```bash
make api
# → http://localhost:8000/docs
```

---

## Demo

The Streamlit UI shows the full pipeline in real time:

- **Step cards** update as each LangGraph node executes
- **Relevance bar chart** shows per-document scores vs threshold
- **Routing badge** indicates whether web search was triggered
- **Source panel** shows content previews with URLs for web sources
- **Raw state inspector** for debugging pipeline internals

Try these queries to see the routing behavior:

| Query | Expected routing |
|---|---|
| "What is the decompose-and-recompose technique in CRAG?" | Local (covered in sample docs) |
| "How does HNSW work in vector databases?" | Local |
| "What did Anthropic announce at their 2025 developer day?" | Web fallback |
| "What is the current pricing for GPT-4o?" | Web fallback |

---

## REST API

```bash
# Health check
curl http://localhost:8000/health

# Query the pipeline
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is corrective RAG?",
    "relevance_threshold": 0.6,
    "top_k": 5
  }'

# Add documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your document text here..."], "metadatas": [{"source": "manual"}]}'
```

Response schema:

```json
{
  "query": "...",
  "answer": "...",
  "web_search_triggered": false,
  "avg_relevance_score": 0.82,
  "sources": [{"source": "local", "score": 0.87, "content_preview": "..."}],
  "pipeline_log": [
    "[RETRIEVE] ChromaDB → 5 documents fetched (top-5)",
    "[EVALUATE] Scores: ['0.91', '0.85', '0.78'] | avg=0.82 (threshold=0.6) ✓ sufficient",
    "[FILTER] Recomposed 14 strips from 5 docs (5 local + 0 web)",
    "[GENERATE] Answer produced using local context"
  ]
}
```

---

## Evaluation

Run the benchmark comparing **Baseline RAG** (retrieve → generate, no self-evaluation) against **Corrective RAG** on 30 questions:

```bash
# Full benchmark (costs ~$2-3 in API calls)
make eval

# Quick sanity check (5 questions)
make eval-quick

# Local knowledge only
make eval-local

# Web-fallback questions only
make eval-web
```

The benchmark uses Claude as a judge scoring each answer on:
- **Faithfulness** — is every claim grounded in context?
- **Relevance** — does the answer address the question?
- **Completeness** — does it cover all key aspects?

Expected results (indicative, varies by corpus):

| Metric | Baseline RAG | Corrective RAG |
|---|---|---|
| Overall avg score | ~0.61 | ~0.78 |
| Local questions | ~0.71 | ~0.79 |
| Web-required questions | ~0.38 | ~0.76 |
| Web search rate | 0% | ~35% |

---

## Key Engineering Decisions

**Parallel relevance scoring** — Documents are evaluated concurrently using `ThreadPoolExecutor` since each LLM call is independent. For 5 documents, this reduces evaluation latency by ~4×.

**Haiku for evaluation, Sonnet for generation** — Using `claude-haiku-4-5` for scoring and filtering cuts API costs by ~10× vs using Sonnet throughout, with negligible quality loss on binary relevance judgments.

**Typed pipeline state** — `RAGState` is a `TypedDict` with `Annotated[list[str], operator.add]` for the `pipeline_log` field, enabling LangGraph to accumulate log entries across nodes without explicit state merging.

**Tenacity retry logic** — Relevance scoring calls are wrapped with exponential backoff to handle transient API rate limits gracefully.

**Per-request threshold override** — The FastAPI `/query` endpoint accepts optional `relevance_threshold` and `top_k` overrides, resetting to defaults after each request. This enables A/B testing different configurations without service restarts.

---

## Project Structure

```
self-evaluating-rag/
├── src/
│   ├── config.py               # Pydantic-Settings config from .env
│   ├── ingestion/
│   │   ├── loader.py           # PDF/text loading + chunking
│   │   └── vectorstore.py      # ChromaDB singleton + embedding model
│   └── pipeline/
│       ├── state.py            # RAGState TypedDict (LangGraph)
│       ├── nodes.py            # All 5 pipeline node functions
│       └── graph.py            # LangGraph DAG assembly
├── api/
│   ├── main.py                 # FastAPI app + lifespan warmup
│   └── schemas.py              # Pydantic request/response models
├── evaluation/
│   ├── test_set.py             # 30-question benchmark set
│   └── benchmark.py           # Baseline vs CRAG comparison
├── data/
│   ├── sample_docs/            # 5 AI/ML knowledge base files
│   └── ingest.py               # CLI ingestion script (rich progress)
├── app.py                      # Streamlit UI (real-time pipeline viz)
├── requirements.txt
├── .env.example
└── Makefile                    # make install / ingest / api / app / eval
```

---

## Extending This System

**Add your own knowledge base**: Drop `.txt`, `.pdf`, or `.md` files into `data/sample_docs/` and run `make ingest`, or use the sidebar file uploader in the Streamlit app.

**Change the embedding model**: Update `EMBEDDING_MODEL` in `.env`. Any `sentence-transformers` model works. Run `make ingest-fresh` to rebuild the index.

**Adjust routing sensitivity**: Lower `RELEVANCE_THRESHOLD` (e.g., 0.4) to rely more on local context; raise it (e.g., 0.8) to trigger web search more aggressively.

**Enable LangSmith tracing**: Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=...` in `.env` for full node-level observability in the LangSmith dashboard.

---

## Tech Stack

Python 3.11 · LangChain 0.3 · LangGraph 0.2 · Anthropic Claude API · ChromaDB · HuggingFace sentence-transformers · Tavily · FastAPI · Streamlit · Plotly · Pydantic v2 · Tenacity
