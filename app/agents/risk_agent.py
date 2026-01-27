from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL
from app.state import AppState

# --- SCHEMAS ---

class RiskEntry(BaseModel):
    """Schema for Agent 3: Generating the technical risk profile."""
    risk_statement: str
    impact: Literal["Low", "Medium", "High"]
    likelihood: Literal["Low", "Medium", "High"]
    rating: Literal["Low", "Medium", "High", "Critical"]
    recommended_control: str

class RiskDecision(BaseModel):
    """Schema for Agent 4: The final materiality sign-off."""
    route: Literal["KEEP_RISK", "DROP_RISK"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str

# --- AGENTS ---

def risk_assessment_agent(state: AppState):
    """
    Agent 3: Risk Expert.
    Analyzes the gap to create a formal risk profile.
    """
    # Context handling based on the Router's path
    gap_context = ""
    if state.gap_route == "KEEP_GAP":
        gap_context = (
            f"\nDetected Gap: {state.gap_summary}\n"
            f"Auditor Severity: {state.gap_severity}\n"
            f"Auditor Recommendation: {state.gap_recommendation}"
        )
    elif state.gap_route == "NO_GAP_HIGH_RISK":
        gap_context = "\nNote: No formal compliance gap identified, but high-risk activity detected."

    prompt = f"""
    You are a Senior Risk Assessment Agent.
    
    Requirement: {state.requirement}
    Evidence: {state.evidence}
    {gap_context}

    TASK:
    1. Draft a concise Risk Statement (Event -> Consequence).
    2. Assess Impact and Likelihood based on the evidence provided.
    3. Determine the final Risk Rating (Low, Medium, High, or Critical).
    4. Suggest a specific technical Control to mitigate this risk.

    Return STRICT JSON matching the RiskEntry schema.
    """

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    # We use the full RiskEntry schema so all state variables get filled
    risk = llm.with_structured_output(RiskEntry).invoke(prompt)

    # Syncing data back to the AppState
    state.risk_statement = risk.risk_statement
    state.impact = risk.impact
    state.likelihood = risk.likelihood
    state.rating = risk.rating
    state.recommended_control = risk.recommended_control
    
    return state


def risk_materiality_agent(state: AppState):
    """
    Agent 4: Chief Risk Officer (Materiality).
    Decides if the risk is significant enough for the final report.
    """
    prompt = f"""
    You are a Risk Materiality Decision Agent. 
    Review the following risk profile to decide if it should be reported to management.

    Risk Profile:
    - Statement: {state.risk_statement}
    - Rating: {state.rating}
    - Impact: {state.impact}
    - Likelihood: {state.likelihood}

    Decision Logic:
    - KEEP_RISK: If the rating is Medium or higher, or has regulatory/operational impact.
    - DROP_RISK: If the risk is purely theoretical, low impact, and low likelihood.

    Return STRICT JSON matching the RiskDecision schema.
    """

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0) # Zero temp for logical consistency
    decision = llm.with_structured_output(RiskDecision).invoke(prompt)

    state.risk_route = decision.route
    state.risk_confidence = decision.confidence
    state.risk_reason = decision.reason
    
    return state