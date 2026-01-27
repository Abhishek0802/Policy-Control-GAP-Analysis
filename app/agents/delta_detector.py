from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)


def delta_detector(requirement: str, internal_policy_text: str) -> str:
    """
    Determines whether a new requirement is already covered by
    existing internal policy text.

    Returns ONLY one of:
    COVERED | NOT_COVERED | STRONGER
    """

    prompt = f"""
You are performing requirement delta analysis.

Internal policy content:
{internal_policy_text}

New requirement:
{requirement}

Decision rules:
- COVERED: Internal policy already addresses this requirement.
- STRONGER: Internal policy covers this requirement with stronger controls.
- NOT_COVERED: Requirement is missing or materially weaker.

Answer with EXACTLY one word:
COVERED or NOT_COVERED or STRONGER
"""

    resp = llm.invoke(prompt).content.strip().upper()
    return resp