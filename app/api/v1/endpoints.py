"""
Bhairava — Fraud Detection System
app/api/v1/endpoints.py

FastAPI router for fraud prediction and auto-responder services.
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.transaction import (
    TransactionPayload,
    FraudPredictionResponse,
    AutoResponseDecision,
)
from app.services.fraud_detector import detector_service
from app.services.auto_responder import auto_responder_service


router = APIRouter(prefix="/api/v1", tags=["Fraud Prevention"])


@router.post(
    "/predict-fraud",
    response_model=FraudPredictionResponse,
    summary="Stage 1: Real-time fraud probability scoring",
    status_code=status.HTTP_200_OK,
)
async def predict_fraud(payload: TransactionPayload):
    """
    Evaluates incoming transaction data against the trained Bhairava XGBoost model.
    Returns estimated risk probability score (0.0 to 1.0) and latency telemetry.
    """
    try:
        response, _ = detector_service.predict_risk(payload)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post(
    "/auto-respond",
    response_model=AutoResponseDecision,
    summary="Stage 2: Policy decisioning and automated action",
    status_code=status.HTTP_200_OK,
)
async def auto_respond(payload: TransactionPayload):
    """
    Executes complete end-to-end evaluation:
    1. Scores transaction risk probability.
    2. Evaluates cost-optimal 3-tier merchant policy (ALLOW, CHALLENGE_3DS, AUTO_DECLINE).
    3. Attributes explicit, human-readable reason codes.
    4. Formulates real-time merchant alert notifications and audit tracking ID.
    """
    try:
        decision = auto_responder_service.evaluate_transaction(payload)
        return decision
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision engine error: {str(e)}",
        )


@router.get(
    "/health",
    summary="Service and model health check",
    status_code=status.HTTP_200_OK,
)
async def health_check():
    """
    Verifies that the API service, feature schema, and trained model artifacts are healthy.
    """
    is_model_loaded = detector_service.model is not None
    num_features = len(detector_service.feature_names)
    locked_threshold = detector_service.locked_threshold

    return {
        "status": "online" if is_model_loaded else "degraded",
        "service": "Bhairava AI — Fraud Prevention Engine",
        "version": "1.0.0",
        "model_loaded": is_model_loaded,
        "feature_count": num_features,
        "locked_decision_threshold": locked_threshold,
    }
