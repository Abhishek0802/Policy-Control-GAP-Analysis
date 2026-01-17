# app/rag/temp_ingest.py
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.llm.openai_client import embeddings

def build_temp_faiss(pdf_path: str) -> FAISS:
    docs = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)
