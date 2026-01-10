from __future__ import annotations
from dataclasses import dataclass
from openai import OpenAI
from app.config import OPENAI_API_KEY, CHAT_MODEL, EMBED_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Chat ----------
def chat(system: str, user: str, temperature: float = 0.2) -> str:
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


@dataclass
class SimpleLLM:
    system: str = "You are a helpful assistant."
    temperature: float = 0.2

    def invoke(self, prompt: str) -> str:
        return chat(self.system, prompt, temperature=self.temperature)


# what agents import
llm = SimpleLLM()


# ---------- Embeddings ----------
class OpenAIEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = _client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        resp = _client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding


# ✅ what RAG code imports
embeddings = OpenAIEmbeddings()
