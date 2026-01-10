from pydantic import BaseModel
from typing import Literal
from app.llm.openai_client import llm

class GapRoute(BaseModel):
    route: Literal["KEEP_GAP", "DROP_GAP"]
    confidence: float
    reason: str

def route_gap(requirement, evidence):
    prompt = f"""
You are a compliance triage agent.

Requirement:
{requirement}

Evidence:
{evidence}

Decide:
- KEEP_GAP if materially affects risk or compliance
- DROP_GAP if cosmetic or non-impactful

Return JSON with:
route, confidence (0-1), reason
"""

    response = llm.with_structured_output(GapRoute).invoke(prompt)
    return response
