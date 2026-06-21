from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_directory(path: str, glob: str = "**/*.{txt,pdf,md}") -> list[Document]:
    loaders = {
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".md": lambda p: TextLoader(p, encoding="utf-8"),
        ".pdf": lambda p: PyPDFLoader(p),
    }

    docs = []
    for file_path in Path(path).rglob("*"):
        suffix = file_path.suffix.lower()
        if suffix in loaders:
            try:
                loader = loaders[suffix](str(file_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: could not load {file_path}: {e}")

    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
