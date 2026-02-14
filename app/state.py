from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal, Annotated
from pydantic import BaseModel, Field
import operator

# Define strict choices for AI decision-making
GapRoute = Literal["KEEP_GAP", "DROP_GAP", "NO_GAP_HIGH_RISK"]
SeverityLevel = Literal["Low", "Medium", "High"]
RiskRating = Literal["Low", "Medium", "High", "Critical"]

class AppState(BaseModel):
    """
    The 'Smart Folder' that travels through the LangGraph agents.
    It carries the policy clause from initial upload to final risk report.
    """

    # --- 1. ENTRY DATA (Set by Flask) ---
    requirement: str                   # The actual text from the user's policy
    evidence: str = ""                 # Supporting text found in the document
    
    # --- 2. TRIAGE DATA (Set by Router Agent) ---
    gap_route: Optional[GapRoute] = None
    gap_reason: Optional[str] = None   # Why the router chose this path

    # --- 3. AUDIT FINDINGS (Set by Gap Agent) ---
    gap_summary: Optional[str] = None  # Explanation of what ISO rule details are missing
    gap_status: Optional[str] = None   # "Fully Meets", "Partially Meets", "Does Not Meet"
    gap_recommendation: Optional[str] = None
    source_ref: Optional[str] = None  # Dynamic reference from the PDF (e.g., "ISO 27001 Annex A.8.10")

    # --- 4. RISK ASSESSMENT (Set by Risk Agents) ---
    risk_statement: Optional[str] = None
    impact: Optional[SeverityLevel] = None
    likelihood: Optional[SeverityLevel] = None
    rating: Optional[RiskRating] = None
    recommended_control: Optional[str] = None

    # --- 5. PERSISTENT STORAGE ---
    audit_log: Annotated[List[Dict[str, Any]], operator.add] = []