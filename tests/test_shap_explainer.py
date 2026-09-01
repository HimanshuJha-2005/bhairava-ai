"""
Bhairava — Fraud Detection System
tests/test_shap_explainer.py

Unit tests for the TreeSHAP local explainability service.
"""

import pytest
from ml.evaluation.shap_explainer import shap_service
from app.services.auto_responder import auto_responder_service
from app.schemas.transaction import TransactionPayload


def test_shap_explainer_computes_contributions():
    """Verify that TreeSHAP computes top features and directions properly."""
    feature_map = {
        "TransactionAmt": 1200.0,
        "amount_decimal": 0.99,
        "card1": 13926,
        "card2": 321.0,
        "card3": 150.0,
        "addr1": 299.0,
        "P_emaildomain": 1,
        "has_identity_data": 0,
    }

    explanation = shap_service.explain_transaction(feature_map, top_k=5)

    assert "base_score" in explanation
    assert "top_features" in explanation
    assert len(explanation["top_features"]) <= 5

    for item in explanation["top_features"]:
        assert "feature" in item
        assert "friendly_name" in item
        assert "shap_value" in item
        assert item["direction"] in ["increases_risk", "reduces_risk"]


def test_auto_responder_attaches_shap():
    """Verify that AutoResponseDecision includes the shap_attribution block."""
    payload = TransactionPayload(
        transaction_id="TXN_TEST_SHAP_01",
        amount=850.0,
        card1=13926,
        card2=321.0,
        card3=150.0,
        addr1=299.0,
        P_emaildomain="gmail.com",
        has_identity_data=1,
    )

    decision = auto_responder_service.evaluate_transaction(payload)

    assert decision.shap_attribution is not None
    assert isinstance(decision.shap_attribution.base_score, float)
    assert len(decision.shap_attribution.top_features) > 0

    first_feat = decision.shap_attribution.top_features[0]
    assert first_feat.direction in ["increases_risk", "reduces_risk"]
