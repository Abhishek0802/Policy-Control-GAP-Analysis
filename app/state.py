from __future__ import annotations
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


GapRoute = Literal["KEEP_GAP", "DROP_GAP", "NO_GAP_HIGH_RISK"]
RiskRoute = Literal["KEEP_RISK", "DROP_RISK"]


class AppState(BaseModel):
    # -------- Inputs --------
    requirement: str
    evidence: str = ""
    scope: str = "Policy & Control Gap Analysis"

    # -------- Router decision (Gap triage) --------
    gap_route: Optional[GapRoute] = None
    gap_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gap_reason: Optional[str] = None

    # -------- Optional gap finding (only if KEEP_GAP) --------
    gap_summary: Optional[str] = None
    gap_severity: Optional[Literal["Low", "Medium", "High"]] = None
    gap_recommendation: Optional[str] = None

    # -------- Risk draft --------
    risk_statement: Optional[str] = None
    impact: Optional[Literal["Low", "Medium", "High"]] = None
    likelihood: Optional[Literal["Low", "Medium", "High"]] = None
    rating: Optional[Literal["Low", "Medium", "High", "Critical"]] = None
    recommended_control: Optional[str] = None

    # -------- Risk materiality --------
    risk_route: Optional[RiskRoute] = None
    risk_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_reason: Optional[str] = None

    # -------- Audit logs --------
    audit_log: List[Dict[str, Any]] = []
