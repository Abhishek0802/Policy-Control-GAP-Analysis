from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

# Define strict choices for AI decision-making
GapRoute = Literal["KEEP_GAP", "DROP_GAP", "NO_GAP_HIGH_RISK"]
RiskRoute = Literal["KEEP_RISK", "DROP_RISK"]
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
    source_ref: str = "N/A"
    
    # --- 2. TRIAGE DATA (Set by Router Agent) ---
    gap_route: Optional[GapRoute] = None
    gap_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gap_reason: Optional[str] = None   # Why the router chose this path

    # --- 3. AUDIT FINDINGS (Set by Gap Agent) ---
    gap_summary: Optional[str] = None  # Explanation of what ISO rule is missing
    gap_severity: Optional[SeverityLevel] = None
    gap_recommendation: Optional[str] = None

    # --- 4. RISK ASSESSMENT (Set by Risk Agents) ---
    risk_statement: Optional[str] = None
    impact: Optional[SeverityLevel] = None
    likelihood: Optional[SeverityLevel] = None
    rating: Optional[RiskRating] = None
    recommended_control: Optional[str] = None

    # --- 5. FINAL DECISION (Set by Materiality Agent) ---
    risk_route: Optional[RiskRoute] = None
    risk_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_reason: Optional[str] = None

    # --- 6. PERSISTENT STORAGE ---
    audit_log: List[Dict[str, Any]] = [] # Final history of all analyzed clauses