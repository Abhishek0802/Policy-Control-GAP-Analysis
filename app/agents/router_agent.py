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

    ### CRITICAL FILTERING RULE:
    If the INTERNAL POLICY REQUIREMENT is an administrative statement, a header, or introductory text 
    (e.g., "Purpose", "Scope", "Applicability", "Version Control", "Policy Owner", "Introduction"), 
    you MUST select 'DROP_GAP'. These are not auditable controls.

    ### INTERNAL POLICY REQUIREMENT:
    "{state.requirement}"

    ### EVIDENCE FOUND:
    "{state.evidence}"

    ### CONTEXTUAL DATA:
    - Current Gap Severity: {getattr(state, 'gap_severity', 'N/A')}
    - Current Risk Rating: {getattr(state, 'rating', 'N/A')}

    ### DECISION LOGIC:
    1. KEEP_GAP: Use if this is a functional requirement/control that has a Medium/High gap OR a clear promise made that MUST be verified against the evidence.
    
    2. NO_GAP_HIGH_RISK: Use if the clause is a functional requirement that looks compliant but involves inherently risky operations (e.g., manual deletion, root access, unmonitored transfers).
    
    3. DROP_GAP: Use if:
       - The text is Administrative (Purpose, Scope, Definitions, etc.).
       - The text is a Header or Footer.
       - The text contains no actionable instructions or controls.
       - The requirement is completely unrelated to the provided evidence.

    ### OUTPUT INSTRUCTIONS:
    Return ONLY STRICT JSON. 
    Ensure the 'gap_status' field reflects the logic: 
    - For DROP_GAP, status should be "Out of Scope" or "Not Applicable".
    - For KEEP_GAP, status should be "Does Not Meet" or "Partially Meets".
    - For NO_GAP_HIGH_RISK, status should be "Fully Meets" (but flagged for risk).
    """

    # Using structured output ensures the AI doesn't return conversational text
    structured_llm = llm.with_structured_output(RouterDecision)
    decision = structured_llm.invoke(prompt)

    # Record the findings back to the 'Smart Clipboard' (AppState)
    state.gap_route = decision.route
    state.gap_reason = decision.reason

    return state