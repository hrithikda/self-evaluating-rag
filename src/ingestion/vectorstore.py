from functools import lru_cache

from langchain_chroma import Chroma

from src.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(documents: list, metadatas: list[dict] | None = None) -> list[str]:
    """Add raw text documents to ChromaDB and return their IDs."""
    from langchain_core.documents import Document

    vs = get_vectorstore()
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(documents, metadatas or [{}] * len(documents))
    ]
    return vs.add_documents(docs)


def collection_size() -> int:
    vs = get_vectorstore()
    return vs._collection.count()
