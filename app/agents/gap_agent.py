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
    Purpose: Compares the internal policy against the specific section of the standard retrieved from the documents.
    """
    
    # Using temperature 0 is vital here for "Grounding" (sticking to the facts)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0) 
    
    prompt = f"""
    You are a Senior ISO Compliance Lead Auditor.

    You MUST base all conclusions ONLY on the RETRIEVED DOCUMENT EVIDENCE.
    Do NOT infer beyond the text.
    Do NOT restate the requirement unless necessary.

    INTERNAL REQUIREMENT:
    "{state.requirement}"

    RETRIEVED DOCUMENT EVIDENCE (Source of Truth):
    "{state.evidence}"

    TASK:

    1. Extract the exact Clause ID or Section Number explicitly mentioned in the RETRIEVED EVIDENCE.
    - If none is present, return "Not Explicitly Stated".

    2. Determine whether the INTERNAL REQUIREMENT:
    - Fully Meets
    - Partially Meets
    - Does Not Meet
    the requirement described in the RETRIEVED EVIDENCE.

    3. gap_summary:
    - Maximum 3 bullet-style sentences.
    - No introductions.
    - No conclusions.
    - No filler language.
    - State only objective comparison findings.

    4. recommendation:
    - Bullet-style action steps written as short imperative statements.
    - No explanation.
    - No narrative justification.

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