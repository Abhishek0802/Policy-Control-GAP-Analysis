from __future__ import annotations

import os
from typing import List, Tuple

from app.config import FAISS_INDEX_PATH
from app.llm.openai_client import embeddings

# If you're using LangChain FAISS vectorstore:
from langchain_community.vectorstores import FAISS


def load_faiss_retriever(k: int = 6):
    """
    Loads a FAISS index from FAISS_INDEX_PATH and returns a retriever.
    Assumes index was saved via FAISS.save_local(FAISS_INDEX_PATH).
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at: {FAISS_INDEX_PATH}. "
            f"Run ingest first to create it."
        )

    # Important: allow_dangerous_deserialization is often required depending on version
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        lambda text: embeddings.embed_query(text),
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_controls(query: str, k: int = 3):
    """
    Convenience helper: loads retriever and returns joined text context.
    """
    retriever = load_faiss_retriever(k=k)
    docs = retriever.get_relevant_documents(query) ## Requirement is passed here

    # Join as context
    chunks = []
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        if text.strip():
            chunks.append(text.strip())

    return docs
