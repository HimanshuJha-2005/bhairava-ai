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
from app.schemas.audit import (
    FeedbackRequest,
    FeedbackResponse,
    TransactionAuditHistory,
    AuditEntry,
    AggregateStats,
    ActionBreakdown,
)
from app.services.fraud_detector import detector_service
from app.services.auto_responder import auto_responder_service
from app.services.audit_store import audit_store


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


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Stage 3: Submit confirmed transaction outcome for a prior decision",
    status_code=status.HTTP_200_OK,
)
async def submit_feedback(payload: FeedbackRequest):
    """
    Accepts a merchant-confirmed outcome label for any previously evaluated
    transaction. Used to close the loop between model decisions and real-world
    outcomes, enabling future model improvement tracking.

    - **fraud_confirmed**: The transaction was verified as fraudulent after the fact.
    - **legitimate_confirmed**: The transaction was a genuine purchase by a real cardholder.
    """
    success = audit_store.log_feedback(payload.audit_id, payload.outcome.value)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No decision found with audit_id '{payload.audit_id}'. "
                   "Ensure the transaction was evaluated via /auto-respond first.",
        )
    return FeedbackResponse(
        audit_id=payload.audit_id,
        outcome=payload.outcome.value,
        message=f"Outcome '{payload.outcome.value}' recorded for audit_id {payload.audit_id}.",
    )


@router.get(
    "/audit/{transaction_id}",
    response_model=TransactionAuditHistory,
    summary="Stage 3: Retrieve full decision audit trail for a transaction",
    status_code=status.HTTP_200_OK,
)
async def get_audit_history(transaction_id: str):
    """
    Returns the complete history of fraud decisions and any associated merchant
    feedback for a given transaction_id. Useful for dispute resolution and
    post-incident forensics.
    """
    records = audit_store.get_decision_history(transaction_id)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No audit records found for transaction_id '{transaction_id}'.",
        )
    entries = [AuditEntry(**r) for r in records]
    return TransactionAuditHistory(
        transaction_id=transaction_id,
        total_decisions=len(entries),
        decisions=entries,
    )


@router.get(
    "/stats",
    response_model=AggregateStats,
    summary="Stage 3: Aggregate operational metrics across all evaluated transactions",
    status_code=status.HTTP_200_OK,
)
async def get_stats():
    """
    Returns live aggregate statistics across all persisted fraud decisions:
    - Total transactions evaluated
    - Action distribution (ALLOW / CHALLENGE_3DS / AUTO_DECLINE)
    - Average risk score across the population
    - Merchant-reported fraud confirmation rate
    """
    raw = audit_store.get_aggregate_stats()
    return AggregateStats(
        total_evaluated=raw["total_evaluated"],
        action_breakdown=ActionBreakdown(**raw["action_breakdown"]),
        avg_risk_score=raw["avg_risk_score"],
        fraud_confirmation_rate=raw["fraud_confirmation_rate"],
        feedback_submitted=raw["feedback_submitted"],
    )
