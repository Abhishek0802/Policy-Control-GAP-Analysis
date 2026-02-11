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
            f"Auditor Recommendation: {state.gap_recommendation}\n"
            f"Auditor Source Reference: {state.source_ref}"
        )
    elif state.gap_route == "NO_GAP_HIGH_RISK":
        gap_context = "\nNote: No formal compliance gap identified, but high-risk activity detected."

    prompt = f"""
    You are a Senior Technical Risk Assessor.

    Use ONLY the information provided.
    Do NOT add narrative explanation.
    Do NOT justify your reasoning.
    Write in short, clinical statements only.

    Requirement:
    {state.requirement}

    Evidence:
    {state.evidence}
    {gap_context}

    TASK RULES:

    1. risk_statement:
    - One sentence only.
    - Format strictly: "If <event>, then <consequence>."
    - Maximum 10 words.
    - No adjectives, no risk commentary.

    2. impact:
    - Choose: Low | Medium | High
    - Base ONLY on operational, financial, or regulatory damage.

    3. likelihood:
    - Choose: Low | Medium | High
    - Base ONLY on evidence strength and control weakness.

    4. rating:
    - Use matrix logic:
        High + High = Critical
        High + Medium = High
        Medium + Medium = Medium
        Anything Low-dominant = Low or Medium
    - Do NOT explain.

    5. recommended_control:
    - Maximum 2 short imperative sentences.
    - Technical action only.
    - No justification.
    - No narrative language.

    Return STRICT JSON matching the RiskEntry schema.
    If any field exceeds limits, you are over-analyzing.
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
    You are a Risk Materiality Decision Engine.

    Apply the reporting threshold strictly.

    Risk Rating: {state.rating}

    DECISION RULES:

    KEEP_RISK:
    - Rating is Medium, High, or Critical

    DROP_RISK:
    - Rating is Low

    Do NOT re-evaluate impact or likelihood.
    Do NOT generate analysis.
    Base the decision ONLY on the rating.

    Return STRICT JSON matching the RiskDecision schema.
    """


    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0) # Zero temp for logical consistency
    decision = llm.with_structured_output(RiskDecision).invoke(prompt)

    state.risk_route = decision.route
    state.risk_confidence = decision.confidence
    state.risk_reason = decision.reason
    
    return state