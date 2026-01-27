from langchain.vectorstores import FAISS
from app.llm.openai_client import embeddings


_VECTOR_DB = None


def get_vector_db():
    global _VECTOR_DB
    if _VECTOR_DB is None:
        _VECTOR_DB = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
        )
    return _VECTOR_DB