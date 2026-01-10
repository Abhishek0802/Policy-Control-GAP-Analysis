from typing import Literal
from pydantic import BaseModel
from app.llm.openai_client import llm


class GapFinding(BaseModel):
    gap_summary: str
    severity: Literal["Low", "Medium", "High"]
    recommendation: str


def gap_agent(state):
    prompt = f"""
You are a compliance gap analyst.

Requirement:
{state.requirement}

Evidence:
{state.evidence}

Create a concise consulting-ready gap finding.

Return STRICT JSON:
{{
  "gap_summary": "...",
  "severity": "Low" | "Medium" | "High",
  "recommendation": "..."
}}
"""
    out = llm.with_structured_output(GapFinding).invoke(prompt)
    state.gap_summary = out.gap_summary
    state.gap_severity = out.severity
    state.gap_recommendation = out.recommendation
    return state
