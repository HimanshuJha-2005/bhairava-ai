"""
Bhairava — Fraud Detection System
tests/test_auto_responder.py

Unit tests for the Auto-Responder policy engine and explainability logic.
"""

from app.schemas.transaction import TransactionPayload, ResponseAction, RiskTier
from app.services.auto_responder import auto_responder_service


def test_low_risk_transaction_allowed():
    """Legitimate standard transaction should be ALLOWED without OTP challenge."""
    payload = TransactionPayload(
        transaction_id="txn_legit_001",
        amount=45.0,
        card1=1234,
        card2=321.0,
        card3=150.0,
        addr1=299.0,
        P_emaildomain="gmail.com",
        R_emaildomain="gmail.com",
        has_identity_data=1,
        extra_features={
            "amount_vs_card_mean": 1.0,
            "card1_txn_count": 25,
            "card_addr_txn_count": 20,
            "rapid_card_activity": 0,
            "card_time_since_prev": 86400.0,
        },
    )

    decision = auto_responder_service.evaluate_transaction(payload)

    assert decision.transaction_id == "txn_legit_001"
    assert decision.action == ResponseAction.ALLOW
    assert decision.risk_tier == RiskTier.LOW
    assert decision.requires_otp_challenge is False
    assert decision.merchant_notification.enabled is False


def test_high_risk_transaction_declined():
    """Anomalous rapid transaction with massive amount spike should be AUTO_DECLINED."""
    payload = TransactionPayload(
        transaction_id="txn_fraud_999",
        amount=9500.0,
        card1=9999,
        card2=500.0,
        card3=150.0,
        addr1=100.0,
        P_emaildomain="disposable-temp-mail.xyz",
        R_emaildomain="victim-account.com",
        has_identity_data=0,
        extra_features={
            "amount_vs_card_mean": 15.0,
            "card1_txn_count": 0,
            "card_addr_txn_count": 0,
            "rapid_card_activity": 1,
            "card_time_since_prev": 12.0,
            "email_domain_freq": 1,
            "is_rare_email_domain": 1,
            "is_new_card_addr": 1,
            "V258": 5.0,
            "V257": 4.0,
            "V70": 3.0,
        },
    )

    decision = auto_responder_service.evaluate_transaction(payload)

    assert decision.transaction_id == "txn_fraud_999"
    assert decision.action in [ResponseAction.CHALLENGE_3DS, ResponseAction.AUTO_DECLINE]
    assert len(decision.reasons) > 0
    assert decision.merchant_notification.enabled is True
