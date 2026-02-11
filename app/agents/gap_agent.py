from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL
from app.state import AppState

class GapFinding(BaseModel):
    """Structured response for the Auditor's findings."""
    gap_summary: str
    severity: Literal["Low", "Medium", "High"]
    recommendation: str
    # This field ensures the AI points to the exact section in your PDFs
    source_ref: str = Field(
        description="The specific clause or section number from the standard, e.g., 'ISO 27001 Annex A.8.10'"
    )

def gap_agent(state: AppState):
    """
    Agent 2: The Auditor
    Purpose: Compares the internal policy against the specific section of the 
    standard retrieved from the documents.
    """
    
    # Using temperature 0 is vital here for "Grounding" (sticking to the facts)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0) 
    
    prompt = f"""
    You are a Senior ISO Compliance Lead Auditor.
    
    AUDIT SCOPE: 
    
    INTERNAL REQUIREMENT TO AUDIT:
    "{state.requirement}"
    
    RETRIEVED DOCUMENT EVIDENCE (Source of Truth):
    "{state.evidence}"
    
    TASK:
    1. Identify the specific Clause ID or Section Number from the RETRIEVED EVIDENCE.
    2. Evaluate if the Internal Requirement satisfies the standard.
    3. Provide a 'gap_summary' as a series of brief, factual sentences (no more than 3-4).
    4. Provide 'recommendation' as specific, actionable steps.

    Return STRICT JSON matching the schema.
    """

    # Structured output forces the LLM to follow the Pydantic model
    analysis = llm.with_structured_output(GapFinding).invoke(prompt)

    # Save findings to the State for the next agent (Risk Agent)
    state.gap_summary = analysis.gap_summary
    state.gap_severity = analysis.severity
    state.gap_recommendation = analysis.recommendation
    state.source_ref = analysis.source_ref  # Dynamic reference from the PDF

    return state