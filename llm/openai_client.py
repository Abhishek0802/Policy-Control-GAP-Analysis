from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import CHAT_MODEL, EMBED_MODEL

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.2
)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL
)
