from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL
from app.rag.retriever import retrieve_evidence

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)

def delta_detector(requirement: str) -> str:
    internal_context = retrieve_evidence(requirement)

    prompt = f"""
Internal policy:
{internal_context}

New requirement:
{requirement}

Is the new requirement already covered?
Answer ONLY one word:
COVERED or NOT_COVERED or STRONGER
"""

    resp = llm.invoke(prompt).content.strip().upper()
    return resp
