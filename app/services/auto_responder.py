"""
Bhairava — Fraud Detection System
app/services/auto_responder.py

Stage 2: Policy and automated response decision engine.
Translates ML risk probabilities and behavioral heuristics into
cost-optimal business actions.
"""

import uuid
from datetime import datetime, timezone
from app.schemas.transaction import (
    TransactionPayload,
    AutoResponseDecision,
    ResponseAction,
    RiskTier,
    NotificationSeverity,
    MerchantNotification,
)
from app.services.fraud_detector import detector_service
from ml.evaluation.explainability import explain_transaction


class AutoResponderService:
    def __init__(
        self,
        allow_threshold: float = 0.35,
        decline_threshold: float = 0.65,
    ):
        self.allow_threshold = allow_threshold
        self.decline_threshold = decline_threshold

    def evaluate_transaction(self, payload: TransactionPayload) -> AutoResponseDecision:
        """
        Executes end-to-end evaluation: ML risk scoring -> Policy decisioning -> Explanation.
        """
        pred_response, feature_map = detector_service.predict_risk(payload)
        base_risk_score = pred_response.risk_score

        # Extract explainability reasons
        reasons = explain_transaction(feature_map, base_risk_score)

        # Multi-layered risk aggregation (Model + Heuristic circuit breakers)
        effective_risk = base_risk_score

        severe_anomalies = {
            "HIGH_GLOBAL_AMOUNT_ANOMALY",
            "HIGH_FREQUENCY_RAPID_CARD_REUSE_DETECTED",
            "TRANSACTION_AMOUNT_UNUSUALLY_HIGH_FOR_CARD",
            "PURCHASER_AND_RECIPIENT_EMAIL_DOMAIN_MISMATCH",
        }
        detected_severe = [r for r in reasons if r in severe_anomalies]

        if len(detected_severe) >= 2:
            effective_risk = max(effective_risk, 0.70)
        elif len(detected_severe) == 1:
            effective_risk = max(effective_risk, 0.35)

        # 3-Tier Policy Decisioning
        if effective_risk < self.allow_threshold:
            action = ResponseAction.ALLOW
            tier = RiskTier.LOW
            requires_otp = False
            notification = MerchantNotification(
                enabled=False,
                severity=NotificationSeverity.INFO,
                title="Transaction Cleared",
                message=f"Transaction {payload.transaction_id} verified as legitimate (Risk: {effective_risk:.2%}).",
                action_required="NONE",
            )

        elif effective_risk < self.decline_threshold:
            action = ResponseAction.CHALLENGE_3DS
            tier = RiskTier.MEDIUM
            requires_otp = True
            notification = MerchantNotification(
                enabled=True,
                severity=NotificationSeverity.WARNING,
                title="Step-Up Verification Triggered",
                message=f"Transaction {payload.transaction_id} routed to 3DS OTP verification due to moderate risk ({effective_risk:.2%}). Reasons: {', '.join(reasons)}",
                action_required="AWAIT_3DS_OTP_AUTHENTICATION",
            )

        else:
            action = ResponseAction.AUTO_DECLINE
            tier = RiskTier.HIGH
            requires_otp = False
            notification = MerchantNotification(
                enabled=True,
                severity=NotificationSeverity.CRITICAL,
                title="High-Risk Fraud Blocked",
                message=f"Transaction {payload.transaction_id} auto-declined to prevent merchant chargeback (Risk: {effective_risk:.2%}). Reasons: {', '.join(reasons)}",
                action_required="LOG_AND_NOTIFY_MERCHANT_SECURITY",
            )

        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        confidence = float(abs(effective_risk - 0.5) * 2.0)

        return AutoResponseDecision(
            transaction_id=payload.transaction_id,
            action=action,
            risk_score=round(effective_risk, 4),
            risk_tier=tier,
            confidence=round(confidence, 4),
            reasons=reasons,
            requires_otp_challenge=requires_otp,
            merchant_notification=notification,
            decision_timestamp=datetime.now(timezone.utc),
            audit_id=audit_id,
        )


# Global singleton
auto_responder_service = AutoResponderService()
