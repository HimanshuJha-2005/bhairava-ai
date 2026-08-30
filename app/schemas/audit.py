"""
Bhairava — Fraud Detection System
app/schemas/audit.py

Pydantic schemas for the Stage 3 audit, feedback, and stats API layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class FeedbackOutcome(str, Enum):
    fraud_confirmed = "fraud_confirmed"
    legitimate_confirmed = "legitimate_confirmed"


class FeedbackRequest(BaseModel):
    audit_id: str = Field(
        ...,
        description="The audit_id returned by a prior /auto-respond call.",
        examples=["aud_bcfb5b41cae7"],
    )
    outcome: FeedbackOutcome = Field(
        ...,
        description="Confirmed real-world outcome for the transaction.",
        examples=["fraud_confirmed"],
    )


class FeedbackResponse(BaseModel):
    audit_id: str
    outcome: str
    message: str


# ---------------------------------------------------------------------------
# Audit history
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    audit_id: str
    action: str
    risk_score: float
    risk_tier: str
    confidence: float
    reasons: list[str]
    requires_otp: bool
    decided_at: str
    outcome: Optional[str] = None
    feedback_at: Optional[str] = None


class TransactionAuditHistory(BaseModel):
    transaction_id: str
    total_decisions: int
    decisions: list[AuditEntry]


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


class ActionBreakdown(BaseModel):
    ALLOW: int
    CHALLENGE_3DS: int
    AUTO_DECLINE: int


class AggregateStats(BaseModel):
    total_evaluated: int
    action_breakdown: ActionBreakdown
    avg_risk_score: float
    fraud_confirmation_rate: Optional[float] = Field(
        None,
        description="Proportion of feedback-confirmed frauds. null if no feedback submitted yet.",
    )
    feedback_submitted: int
