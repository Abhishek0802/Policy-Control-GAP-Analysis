from typing import Literal
from pydantic import BaseModel, Field
from app.llm.openai_client import llm


class RiskEntry(BaseModel):
    risk_statement: str
    impact: Literal["Low", "Medium", "High"]
    likelihood: Literal["Low", "Medium", "High"]
    rating: Literal["Low", "Medium", "High", "Critical"]
    recommended_control: str


class RiskDecision(BaseModel):
    route: Literal["KEEP_RISK", "DROP_RISK"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


def risk_assessment_agent(state):
    # For NO_GAP_HIGH_RISK, we still draft a risk without claiming a gap exists.
    gap_context = ""
    if state.gap_route == "KEEP_GAP":
        gap_context = f"\nGap:\n{state.gap_summary}\nSeverity:{state.gap_severity}\nRecommendation:{state.gap_recommendation}\n"
    elif state.gap_route == "NO_GAP_HIGH_RISK":
        gap_context = "\nNote: No clear gap identified, but contextual risk indicators suggest elevated risk.\n"

    prompt = f"""
You are a Risk Assessment Agent.

Requirement:
{state.requirement}

Evidence:
{state.evidence}
{gap_context}

Draft a risk register entry (consulting style).

Return STRICT JSON:
{{
  "risk_statement": "...",
  "impact": "Low" | "Medium" | "High",
  "likelihood": "Low" | "Medium" | "High",
  "rating": "Low" | "Medium" | "High" | "Critical",
  "recommended_control": "..."
}}
"""
    risk = llm.with_structured_output(RiskEntry).invoke(prompt)

    state.risk_statement = risk.risk_statement
    state.impact = risk.impact
    state.likelihood = risk.likelihood
    state.rating = risk.rating
    state.recommended_control = risk.recommended_control
    return state


def risk_materiality_agent(state):
    prompt = f"""
You are a Risk Materiality Decision Agent.

Decide if this risk should be reported to client.

Risk:
- Statement: {state.risk_statement}
- Impact: {state.impact}
- Likelihood: {state.likelihood}
- Rating: {state.rating}

Rules:
- KEEP_RISK if decision-relevant (regulatory/financial/operational) or needs action.
- DROP_RISK if low impact AND low likelihood and not worth management attention.

Return STRICT JSON:
{{
  "route": "KEEP_RISK" | "DROP_RISK",
  "confidence": 0 to 1,
  "reason": "one short sentence"
}}
"""
    decision = llm.with_structured_output(RiskDecision).invoke(prompt)

    state.risk_route = decision.route
    state.risk_confidence = decision.confidence
    state.risk_reason = decision.reason
    return state
