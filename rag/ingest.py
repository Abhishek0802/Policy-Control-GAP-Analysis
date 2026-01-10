import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.llm.openai_client import embeddings

def build_faiss_index(texts, save_path="data/faiss_index"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.create_documents(texts)
    db = FAISS.from_documents(docs, embeddings)

    db.save_local(save_path)
    return db
