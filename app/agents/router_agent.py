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

Goal: Decide what to do with the requirement based on evidence and context.

Engagement scope: {state.scope}

Requirement:
{state.requirement}

Evidence (policy/framework snippets):
{state.evidence}

Decide exactly ONE route:
1) KEEP_GAP:
   - There is a meaningful control gap worth reporting.
2) DROP_GAP:
   - Finding is cosmetic/redundant/out-of-scope and not decision-relevant.
3) NO_GAP_HIGH_RISK:
   - No clear compliance gap, but contextual risk is still high and must be evaluated (e.g., threat environment, operational weakness).

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
