"""
Bhairava — Fraud Detection System
tests/test_api.py

Integration tests for FastAPI endpoints: /health, /predict-fraud, and /auto-respond.
"""

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["feature_count"] > 400


def test_predict_fraud_endpoint():
    payload = {
        "transaction_id": "test_txn_001",
        "amount": 120.50,
        "card1": 13926,
        "card2": 321.0,
        "card3": 150.0,
        "addr1": 315.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "has_identity_data": 1,
    }

    response = client.post("/api/v1/predict-fraud", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] < 500  # Strict <500ms requirement


def test_auto_respond_endpoint():
    payload = {
        "transaction_id": "test_txn_002",
        "amount": 4500.00,
        "card1": 99999,
        "card2": 500.0,
        "card3": 150.0,
        "addr1": 100.0,
        "P_emaildomain": "suspicious-proxy.net",
        "R_emaildomain": "target.org",
        "has_identity_data": 0,
        "extra_features": {
            "amount_vs_card_mean": 8.0,
            "rapid_card_activity": 1,
            "card_time_since_prev": 30.0,
        },
    }

    response = client.post("/api/v1/auto-respond", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in ["ALLOW", "CHALLENGE_3DS", "AUTO_DECLINE"]
    assert "reasons" in data
    assert isinstance(data["reasons"], list)
    assert "merchant_notification" in data
    assert "audit_id" in data
