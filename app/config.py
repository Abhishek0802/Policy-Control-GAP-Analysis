import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index")
SCOPE = "Policy & Control GAP Analysis"
