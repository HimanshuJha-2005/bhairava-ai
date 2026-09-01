"""
Bhairava — Fraud Detection System
app/schemas/transaction.py

Pydantic schemas for transaction ingestion, risk scoring,
SHAP explainability, and automated response decisions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResponseAction(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE_3DS = "CHALLENGE_3DS"
    AUTO_DECLINE = "AUTO_DECLINE"


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NotificationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TransactionPayload(BaseModel):
    """
    Standard transaction payload sent by merchant or payment gateway.
    """
    transaction_id: str = Field(..., description="Unique transaction ID from merchant/gateway")
    amount: float = Field(..., gt=0, description="Transaction amount in local currency (e.g. INR/USD)")
    merchant_id: Optional[str] = Field(None, description="Identifier for merchant account")
    customer_id: Optional[str] = Field(None, description="Identifier for customer profile")
    card1: int = Field(..., description="Issuer Identification Number / Bank BIN")
    card2: Optional[float] = Field(None, description="Card category code")
    card3: Optional[float] = Field(None, description="Card country code")
    card4: Optional[str] = Field(None, description="Card brand (visa, mastercard, discover, etc.)")
    card5: Optional[float] = Field(None, description="Card type code")
    card6: Optional[str] = Field(None, description="Card funding type (credit, debit)")
    addr1: Optional[float] = Field(None, description="Billing region / zip code prefix")
    addr2: Optional[float] = Field(None, description="Billing country code")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain (e.g. gmail.com)")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    device_type: Optional[str] = Field(None, description="Device category (desktop, mobile)")
    has_identity_data: Optional[int] = Field(1, description="1 if browser/device telemetry is present, 0 otherwise")
    timestamp: Optional[datetime] = Field(default_factory=get_utc_now)

    # Optional extra Vesta features if provided by gateway
    extra_features: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pre-computed gateway features (V1-V338, C1-C14, D1-D15)")


class FraudPredictionResponse(BaseModel):
    """
    Stage 1: Fast ML risk scoring output.
    """
    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Estimated fraud probability")
    risk_tier: RiskTier
    confidence: float = Field(..., ge=0.0, le=1.0)
    inference_time_ms: float
    model_version: str = "bhairava-xgboost-v1.0"


class ShapFeatureContribution(BaseModel):
    """
    Individual feature Shapley contribution.
    """
    feature: str
    friendly_name: str
    shap_value: float
    direction: str = Field(..., description="'increases_risk' or 'reduces_risk'")


class ShapAttributionBlock(BaseModel):
    """
    TreeSHAP explanation payload.
    """
    base_score: float
    top_features: List[ShapFeatureContribution] = Field(default_factory=list)


class MerchantNotification(BaseModel):
    """
    Automated notification dispatched to merchant systems.
    """
    enabled: bool
    severity: NotificationSeverity
    title: str
    message: str
    action_required: str
    timestamp: datetime = Field(default_factory=get_utc_now)


class AutoResponseDecision(BaseModel):
    """
    Stage 2: Full automated decision output.
    """
    transaction_id: str
    action: ResponseAction
    risk_score: float
    risk_tier: RiskTier
    confidence: float
    reasons: List[str]
    shap_attribution: Optional[ShapAttributionBlock] = None
    requires_otp_challenge: bool
    merchant_notification: MerchantNotification
    decision_timestamp: datetime = Field(default_factory=get_utc_now)
    audit_id: str
