from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL


class GapFinding(BaseModel):
    gap_summary: str
    severity: Literal["Low", "Medium", "High"]
    recommendation: str


def gap_agent(state):
    prompt = """
You are a compliance gap analyst.


Requirement:
""" + state.requirement + """


Evidence:
""" + state.evidence + """


Determine ONLY whether a gap exists and its severity.
DO NOT write explanations or recommendations.


Return STRICT JSON:
{
"severity": "Low" | "Medium" | "High"
}
"""
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.2)
    out = llm.with_structured_output(GapFinding).invoke(prompt)
    state.gap_summary = out.gap_summary
    state.gap_severity = out.severity
    state.gap_recommendation = out.recommendation
    return state
