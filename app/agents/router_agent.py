from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL


class RouterDecision(BaseModel):
    route: Literal["KEEP_GAP", "DROP_GAP", "NO_GAP_HIGH_RISK"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


def router_agent(state):
    prompt = f"""
You are a Big-4 style Router/Triage Agent.

You are given ONE control and a prior evaluation result (FULL, PARTIAL, GAP).

Engagement scope: {state.scope}

Requirement:
{state.requirement}

Evidence (policy/framework snippets):
{state.evidence}

Gap severity (if any): {state.gap_severity}
Risk rating (if any): {state.rating}
Control type: {getattr(state, "control_type", "UNKNOWN")}

Decide exactly ONE route:
If gap_severity is Medium or High → KEEP_GAP
- If no gap exists AND risk rating is High or Critical → NO_GAP_HIGH_RISK
- DROP_GAP ONLY if the control is clearly out of scope or intent-only

Return STRICT JSON:
{{
  "route": "KEEP_GAP" | "DROP_GAP" | "NO_GAP_HIGH_RISK",
  "confidence": 0 to 1,
  "reason": "one short sentence"
}}
"""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    decision = llm.with_structured_output(RouterDecision).invoke(prompt)
    state.gap_route = decision.route
    state.gap_confidence = decision.confidence
    state.gap_reason = decision.reason
    return state
