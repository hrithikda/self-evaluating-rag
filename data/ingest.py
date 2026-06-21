#!/usr/bin/env python3
"""
Ingest sample documents into ChromaDB.

Usage:
    python data/ingest.py                        # loads data/sample_docs/
    python data/ingest.py --dir path/to/docs     # custom directory
    python data/ingest.py --clear                # wipe collection first
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import settings
from src.ingestion.loader import load_directory, split_documents
from src.ingestion.vectorstore import add_documents, collection_size, get_vectorstore

console = Console()


def clear_collection() -> None:
    vs = get_vectorstore()
    vs._collection.delete(where={"source": {"$ne": "___never___"}})
    console.print("[yellow]Collection cleared.[/yellow]")


def ingest(source_dir: str) -> None:
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading documents…", total=None)
        raw_docs = load_directory(source_dir)
        progress.update(task, description=f"Loaded {len(raw_docs)} raw documents")

        task2 = progress.add_task("Splitting into chunks…", total=None)
        chunks = split_documents(raw_docs)
        progress.update(task2, description=f"Split into {len(chunks)} chunks")

        task3 = progress.add_task("Embedding + indexing into ChromaDB…", total=None)
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = add_documents(texts, metadatas)
        progress.update(task3, description=f"Indexed {len(ids)} chunks")

    table = Table(title="Ingestion Summary", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="green")
    table.add_row("Source directory", source_dir)
    table.add_row("Raw documents", str(len(raw_docs)))
    table.add_row("Chunks created", str(len(chunks)))
    table.add_row("Chunks indexed", str(len(ids)))
    table.add_row("Collection total", str(collection_size()))
    table.add_row("Embedding model", settings.embedding_model)
    table.add_row("Vector store", settings.chroma_persist_dir)

    console.print(table)
    console.print("[bold green]✓ Ingestion complete![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).parent / "sample_docs"),
        help="Directory containing .txt/.pdf/.md files",
    )
    parser.add_argument("--clear", action="store_true", help="Clear collection before ingesting")
    args = parser.parse_args()

    if args.clear:
        clear_collection()

    ingest(args.dir)


if __name__ == "__main__":
    main()
