from pydantic import BaseModel
from typing import Literal
from app.llm.openai_client import llm

class RiskDecision(BaseModel):
    route: Literal["KEEP_RISK", "DROP_RISK"]
    confidence: float
    reason: str

def route_risk(risk_text):
    prompt = f"""
Decide if this risk should be reported.

Risk:
{risk_text}

KEEP_RISK if decision relevant.
DROP_RISK if low impact & likelihood.
"""

    return llm.with_structured_output(RiskDecision).invoke(prompt)
