from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL
from app.state import AppState

class RouterDecision(BaseModel):
    """Structured output for the Triage decision."""
    route: Literal["KEEP_GAP", "DROP_GAP", "NO_GAP_HIGH_RISK"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str

def router_agent(state: AppState):
    """
    Agent 1: The Inspector (Triage)
    Purpose: Quickly determines if a clause is worth a deep-dive audit 
    based on the specific document evidence provided.
    """
    
    # We use a very low temperature (0) for the Router to ensure 
    # consistent, non-creative categorization.
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    
    prompt = f"""
    You are a Senior Big-4 Compliance Auditor. 
    Your job is to TRIAGE a policy clause against a specific regulatory scope.


    INTERNAL POLICY REQUIREMENT:
    "{state.requirement}"

    EVIDENCE FOUND:
    "{state.evidence}"

    CONTEXTUAL DATA:
    - Current Gap Severity: {getattr(state, 'gap_severity', 'N/A')}
    - Current Risk Rating: {getattr(state, 'rating', 'N/A')}

    DECISION LOGIC:
    1. KEEP_GAP: Use if there's a Medium/High gap OR a clear promise made that MUST be verified against {state.evidence}.
    2. NO_GAP_HIGH_RISK: Use if the clause looks okay but involves dangerous operations (e.g. manual deletion, admin access).
    3. DROP_GAP: Use ONLY if the text is out of scope (e.g. headers, footer, or unrelated to {state.evidence}).

    Return ONLY STRICT JSON matching the schema.
    """

    # Using structured output ensures the AI doesn't return conversational text
    structured_llm = llm.with_structured_output(RouterDecision)
    decision = structured_llm.invoke(prompt)

    # Record the findings back to the 'Smart Clipboard' (AppState)
    state.gap_route = decision.route
    state.gap_confidence = decision.confidence
    state.gap_reason = decision.reason

    return state