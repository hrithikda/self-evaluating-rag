"""
Benchmark: Baseline RAG vs Corrective RAG (CRAG).

Runs the test set through both pipelines and scores each answer using
Claude as a judge on three dimensions:
  - faithfulness  : Is the answer grounded in the provided context?
  - relevance     : Does the answer actually address the question?
  - completeness  : Does the answer cover the key aspects of the question?

Usage:
    python -m evaluation.benchmark                        # full run
    python -m evaluation.benchmark --subset local         # local-only questions
    python -m evaluation.benchmark --subset web           # web-fallback questions
    python -m evaluation.benchmark --limit 5              # quick sanity check
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic

from evaluation.test_set import TEST_QUESTIONS
from src.config import settings
from src.ingestion.vectorstore import collection_size
from src.pipeline.graph import build_graph, make_initial_state

console = Console()


# ---------------------------------------------------------------------------
# LLM judge for answer quality
# ---------------------------------------------------------------------------


class AnswerScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0, description="Grounded in context, no hallucinations")
    relevance: float = Field(ge=0.0, le=1.0, description="Directly addresses the question")
    completeness: float = Field(ge=0.0, le=1.0, description="Covers key aspects of the question")
    reasoning: str = Field(description="One-sentence explanation")


def judge_answer(question: str, context: str, answer: str) -> AnswerScore:
    judge = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.0,
        anthropic_api_key=settings.anthropic_api_key,
    ).with_structured_output(AnswerScore)

    prompt = f"""You are an impartial AI evaluator. Score the following answer on three dimensions:

Question: {question}

Context used:
{context[:2000]}

Answer:
{answer}

Score each dimension 0.0–1.0:
- faithfulness: Is every claim in the answer supported by the context? (1.0 = fully grounded)
- relevance: Does the answer directly address the question? (1.0 = fully on-point)
- completeness: Does the answer cover all key aspects of the question? (1.0 = comprehensive)"""

    return judge.invoke(prompt)


# ---------------------------------------------------------------------------
# Baseline RAG (no correction — just retrieve + generate, no evaluation step)
# ---------------------------------------------------------------------------


def run_baseline(query: str) -> dict:
    from src.ingestion.vectorstore import get_vectorstore

    vs = get_vectorstore()
    try:
        results = vs.similarity_search(query, k=settings.top_k)
    except Exception:
        results = []

    context = "\n\n".join(doc.page_content for doc in results[:3])

    gen_llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0.1,
        anthropic_api_key=settings.anthropic_api_key,
    )

    prompt = (
        f"Answer the following question using the provided context.\n\n"
        f"Question: {query}\n\nContext:\n{context[:4000]}\n\nAnswer:"
    )

    try:
        response = gen_llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Error: {e}"

    return {"answer": answer, "context": context, "web_triggered": False}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(questions: list[dict], output_path: str = "evaluation/results.json") -> None:
    if collection_size() == 0:
        console.print(
            "[bold red]ChromaDB is empty! Run `python data/ingest.py` first.[/bold red]"
        )
        sys.exit(1)

    crag_graph = build_graph()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark…", total=len(questions))

        for q in questions:
            qid = q["id"]
            question = q["question"]
            category = q["category"]

            progress.update(task, description=f"[{qid}] {question[:50]}…")

            # --- Baseline ---
            t0 = time.perf_counter()
            baseline_out = run_baseline(question)
            baseline_time = time.perf_counter() - t0

            # --- CRAG ---
            t0 = time.perf_counter()
            crag_state = crag_graph.invoke(make_initial_state(question))
            crag_time = time.perf_counter() - t0

            # --- Judge ---
            baseline_score = judge_answer(question, baseline_out["context"], baseline_out["answer"])
            crag_score = judge_answer(question, crag_state["filtered_context"], crag_state["answer"])

            results.append(
                {
                    "id": qid,
                    "question": question,
                    "category": category,
                    "baseline": {
                        "answer": baseline_out["answer"][:500],
                        "latency_s": round(baseline_time, 2),
                        "faithfulness": baseline_score.faithfulness,
                        "relevance": baseline_score.relevance,
                        "completeness": baseline_score.completeness,
                        "avg_score": round(
                            (baseline_score.faithfulness + baseline_score.relevance + baseline_score.completeness) / 3,
                            3,
                        ),
                    },
                    "crag": {
                        "answer": crag_state["answer"][:500],
                        "latency_s": round(crag_time, 2),
                        "web_triggered": crag_state["web_search_triggered"],
                        "avg_relevance_score": round(crag_state["avg_relevance"], 3),
                        "faithfulness": crag_score.faithfulness,
                        "relevance": crag_score.relevance,
                        "completeness": crag_score.completeness,
                        "avg_score": round(
                            (crag_score.faithfulness + crag_score.relevance + crag_score.completeness) / 3,
                            3,
                        ),
                    },
                }
            )

            progress.advance(task)

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)
    console.print(f"\n[dim]Full results saved to {output_path}[/dim]")


def _print_summary(results: list[dict]) -> None:
    df = pd.DataFrame(
        [
            {
                "id": r["id"],
                "category": r["category"],
                "baseline_avg": r["baseline"]["avg_score"],
                "crag_avg": r["crag"]["avg_score"],
                "web_triggered": r["crag"]["web_triggered"],
                "delta": r["crag"]["avg_score"] - r["baseline"]["avg_score"],
            }
            for r in results
        ]
    )

    table = Table(title="Benchmark Summary: Baseline RAG vs Corrective RAG", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Baseline RAG", style="yellow")
    table.add_column("Corrective RAG", style="green")
    table.add_column("Delta", style="bold")

    metrics = [
        ("Overall avg score", df["baseline_avg"].mean(), df["crag_avg"].mean()),
    ]

    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        metrics.append(
            (f"  {cat}", sub["baseline_avg"].mean(), sub["crag_avg"].mean())
        )

    for label, b, c in metrics:
        delta = c - b
        delta_str = f"[green]+{delta:.3f}[/green]" if delta > 0 else f"[red]{delta:.3f}[/red]"
        table.add_row(label, f"{b:.3f}", f"{c:.3f}", delta_str)

    web_rate = df["web_triggered"].mean() * 100
    table.add_row("Web search rate", "—", f"{web_rate:.0f}%", "")
    table.add_row("Total questions", str(len(results)), str(len(results)), "")

    console.print("\n")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument("--subset", choices=["local", "web", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()

    questions = TEST_QUESTIONS

    if args.subset == "local":
        questions = [q for q in questions if q["category"] == "answerable_local"]
    elif args.subset == "web":
        questions = [q for q in questions if q["category"] == "requires_web"]

    if args.limit:
        questions = questions[: args.limit]

    console.print(f"[bold]Running {len(questions)} questions (subset={args.subset})[/bold]")
    run_benchmark(questions, args.output)


if __name__ == "__main__":
    main()
