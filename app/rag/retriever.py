from langchain_community.vectorstores import FAISS
from app.llm.openai_client import embeddings

def load_retriever(path="data/faiss_index", k=5):
    db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": k})
