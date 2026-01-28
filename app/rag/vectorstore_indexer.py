import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.llm.openai_client import embeddings

_VECTOR_DB = None

def _load_docs_from_folder(folder: str) -> List[str]:
    texts: List[str] = []
    p = Path(folder)
    if not p.exists():
        return texts

    for file in p.rglob("*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs = loader.load()
            texts.extend([d.page_content for d in docs])
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            docs = loader.load()
            texts.extend([d.page_content for d in docs])

    return texts


def build_faiss_from_folder(folder: str, save_path: str):
    texts = _load_docs_from_folder(folder)
    if not texts:
        raise ValueError(f"No PDF/TXT documents found in: {folder}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.create_documents(texts)

    db = FAISS.from_documents(docs, embeddings)
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)
    return db


def build_temp_faiss(pdf_path: str) -> FAISS:
    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)

def get_vector_db():
    global _VECTOR_DB
    if _VECTOR_DB is None:
        _VECTOR_DB = FAISS.load_local(
            "data/faiss_index", 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        
        _VECTOR_DB.embedding_function = embeddings.embed_query
        
    return _VECTOR_DB