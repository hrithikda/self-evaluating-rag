"""
Benchmark test set: 40 questions spanning local knowledge and web-required knowledge.

Structure:
  - "answerable_local"  — should be answered from the sample docs
  - "requires_web"      — recent events / out-of-corpus facts; should trigger web fallback
"""

TEST_QUESTIONS: list[dict] = [
    # ---- Answerable from local knowledge base (sample_docs) ----
    {
        "id": "L001",
        "question": "What is the difference between supervised and unsupervised learning?",
        "category": "answerable_local",
        "topic": "ai_fundamentals",
    },
    {
        "id": "L002",
        "question": "How does backpropagation work in neural networks?",
        "category": "answerable_local",
        "topic": "ai_fundamentals",
    },
    {
        "id": "L003",
        "question": "What problem did the transformer architecture solve compared to RNNs?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L004",
        "question": "What is Rotary Position Embedding (RoPE) and why is it used?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L005",
        "question": "Explain the decompose-and-recompose technique in Corrective RAG.",
        "category": "answerable_local",
        "topic": "rag_systems",
    },
    {
        "id": "L006",
        "question": "How does HNSW work for approximate nearest neighbor search?",
        "category": "answerable_local",
        "topic": "vector_databases",
    },
    {
        "id": "L007",
        "question": "What is the difference between ChromaDB and Pinecone?",
        "category": "answerable_local",
        "topic": "vector_databases",
    },
    {
        "id": "L008",
        "question": "What is vLLM and how does PagedAttention improve throughput?",
        "category": "answerable_local",
        "topic": "ml_engineering",
    },
    {
        "id": "L009",
        "question": "What metrics does RAGAS use to evaluate RAG systems?",
        "category": "answerable_local",
        "topic": "rag_systems",
    },
    {
        "id": "L010",
        "question": "What is Constitutional AI and how does Anthropic use it?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L011",
        "question": "How does Mixture of Experts (MoE) architecture work?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L012",
        "question": "What is HyDE and how does it improve RAG retrieval?",
        "category": "answerable_local",
        "topic": "rag_systems",
    },
    {
        "id": "L013",
        "question": "What is RLHF and how is it used to align language models?",
        "category": "answerable_local",
        "topic": "ai_fundamentals",
    },
    {
        "id": "L014",
        "question": "How does LSTMs solve the vanishing gradient problem?",
        "category": "answerable_local",
        "topic": "ai_fundamentals",
    },
    {
        "id": "L015",
        "question": "What is Direct Preference Optimization (DPO)?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L016",
        "question": "What are the advantages of speculative decoding for LLM inference?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L017",
        "question": "What is Grouped Query Attention (GQA) and why does it help?",
        "category": "answerable_local",
        "topic": "llm_architectures",
    },
    {
        "id": "L018",
        "question": "How does a feature store improve ML production systems?",
        "category": "answerable_local",
        "topic": "ml_engineering",
    },
    {
        "id": "L019",
        "question": "What is parent-document retrieval in RAG?",
        "category": "answerable_local",
        "topic": "rag_systems",
    },
    {
        "id": "L020",
        "question": "Explain the Self-RAG approach to retrieval-augmented generation.",
        "category": "answerable_local",
        "topic": "rag_systems",
    },

    # ---- Requires web search (recent/out-of-corpus information) ----
    {
        "id": "W001",
        "question": "What new features were announced in the latest OpenAI GPT release?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W002",
        "question": "What is the current pricing for Claude claude-sonnet-4-6 per million tokens?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W003",
        "question": "What papers were accepted at NeurIPS 2024?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W004",
        "question": "What is the current benchmark leaderboard ranking on MMLU?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W005",
        "question": "What are the latest features in LangChain 0.3?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W006",
        "question": "What is Google's Gemini 2.0 Flash model capable of?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W007",
        "question": "What did Anthropic announce about Claude at their latest conference?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W008",
        "question": "What is Meta's Llama 3.3 model and what are its capabilities?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W009",
        "question": "What are the current AWS GPU instance types available for AI workloads?",
        "category": "requires_web",
        "topic": "current_events",
    },
    {
        "id": "W010",
        "question": "What is the latest version of ChromaDB and what changed?",
        "category": "requires_web",
        "topic": "current_events",
    },
]
