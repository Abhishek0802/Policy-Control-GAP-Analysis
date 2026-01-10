# app/state.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class AppState(BaseModel):
    # ---------- Inputs ----------
    # If you’re processing one requirement at a time:
    requirement: Optional[str] = None

    # Evidence text retrieved from FAISS (top-k chunks concatenated)
    evidence: Optional[str] = None

    # Optional: engagement context (helps router decide materiality)
    scope: str = "General compliance review"

    # ---------- Router decisions ----------
    gap_route: Optional[Literal["KEEP_GAP", "DROP_GAP"]] = None
    gap_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gap_reason: Optional[str] = None

    risk_route: Optional[Literal["KEEP_RISK", "DROP_RISK"]] = None
    risk_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_reason: Optional[str] = None

    # ---------- Gap analysis output ----------
    gap_summary: Optional[str] = None
    gap_severity: Optional[Literal["Low", "Medium", "High"]] = None
    gap_recommendation: Optional[str] = None

    # ---------- Risk register output ----------
    risk_statement: Optional[str] = None
    impact: Optional[Literal["Low", "Medium", "High"]] = None
    likelihood: Optional[Literal["Low", "Medium", "High"]] = None
    rating: Optional[Literal["Low", "Medium", "High", "Critical"]] = None
    recommended_control: Optional[str] = None

    # ---------- Logs / final outputs ----------
    dropped_gaps: List[Dict[str, Any]] = []
    dropped_risks: List[Dict[str, Any]] = []
    kept_risks: List[Dict[str, Any]] = []

    def reset_per_item_fields(self) -> None:
        """Call this after processing one requirement to avoid leaking state into the next."""
        self.gap_route = None
        self.gap_confidence = None
        self.gap_reason = None

        self.gap_summary = None
        self.gap_severity = None
        self.gap_recommendation = None

        self.risk_statement = None
        self.impact = None
        self.likelihood = None
        self.rating = None
        self.recommended_control = None

        self.risk_route = None
        self.risk_confidence = None
        self.risk_reason = None
