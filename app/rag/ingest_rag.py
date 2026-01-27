'''
This script ingests internal policy documents from a specified directory, 
constructs a FAISS index/embeddings for efficient retrieval during policy gap analysis, 
and saves the index to a designated path

Ran manually from terminal 'python -m app.rag.ingest_rag.py' to create faiss_index files
'''

from app.rag.ingest_embeddings import build_faiss_from_folder


build_faiss_from_folder(
folder="data/internal_policies",
save_path="data/faiss_index"
)