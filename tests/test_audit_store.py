"""
Bhairava — Fraud Detection System
tests/test_audit_store.py

Unit tests for the SQLite audit store:
  - Decision persistence
  - Feedback ingestion (valid and invalid audit_id)
  - Aggregate stats computation
  - Idempotent duplicate inserts
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from app.services.audit_store import AuditStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    """Isolated in-memory-style SQLite store in a temp directory per test."""
    return AuditStore(db_path=tmp_path / "test_audit.db")


def _decision(store: AuditStore, audit_id: str, txn_id: str, action: str, score: float):
    store.log_decision(
        audit_id=audit_id,
        transaction_id=txn_id,
        action=action,
        risk_score=score,
        risk_tier="LOW" if score < 0.35 else ("MEDIUM" if score < 0.65 else "HIGH"),
        confidence=round(abs(score - 0.5) * 2, 4),
        reasons=["LOW_RISK_NORMAL_BEHAVIOR"] if score < 0.35 else ["HIGH_RISK_SIGNAL"],
        requires_otp=(0.35 <= score < 0.65),
        decided_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_log_and_retrieve_single_decision(store):
    _decision(store, "aud_001", "TXN_A", "ALLOW", 0.10)
    history = store.get_decision_history("TXN_A")
    assert len(history) == 1
    assert history[0]["action"] == "ALLOW"
    assert history[0]["risk_score"] == 0.10
    assert history[0]["requires_otp"] is False


def test_duplicate_audit_id_is_ignored(store):
    """INSERT OR IGNORE — same audit_id twice should not error or duplicate."""
    _decision(store, "aud_dup", "TXN_B", "AUTO_DECLINE", 0.90)
    _decision(store, "aud_dup", "TXN_B", "AUTO_DECLINE", 0.90)
    history = store.get_decision_history("TXN_B")
    assert len(history) == 1


def test_feedback_valid_audit_id(store):
    _decision(store, "aud_fb01", "TXN_C", "AUTO_DECLINE", 0.85)
    result = store.log_feedback("aud_fb01", "fraud_confirmed")
    assert result is True

    history = store.get_decision_history("TXN_C")
    assert history[0]["outcome"] == "fraud_confirmed"
    assert history[0]["feedback_at"] is not None


def test_feedback_invalid_audit_id_returns_false(store):
    result = store.log_feedback("aud_nonexistent", "fraud_confirmed")
    assert result is False


def test_multiple_feedback_entries_for_same_audit_id(store):
    """A single audit_id can have multiple feedback entries (corrections accepted)."""
    _decision(store, "aud_multi", "TXN_D", "CHALLENGE_3DS", 0.50)
    store.log_feedback("aud_multi", "legitimate_confirmed")
    store.log_feedback("aud_multi", "fraud_confirmed")
    history = store.get_decision_history("TXN_D")
    # Two rows due to LEFT JOIN — both feedback entries returned
    assert len(history) == 2


def test_aggregate_stats_empty_store(store):
    stats = store.get_aggregate_stats()
    assert stats["total_evaluated"] == 0
    assert stats["avg_risk_score"] == 0.0
    assert stats["fraud_confirmation_rate"] is None
    assert stats["feedback_submitted"] == 0


def test_aggregate_stats_populated(store):
    for i, (action, score) in enumerate(
        [("ALLOW", 0.10), ("CHALLENGE_3DS", 0.50), ("AUTO_DECLINE", 0.85)]
    ):
        _decision(store, f"aud_agg{i:03d}", f"TXN_S{i}", action, score)

    # Submit feedback for the AUTO_DECLINE only
    store.log_feedback("aud_agg002", "fraud_confirmed")

    stats = store.get_aggregate_stats()
    assert stats["total_evaluated"] == 3
    assert stats["action_breakdown"]["ALLOW"] == 1
    assert stats["action_breakdown"]["CHALLENGE_3DS"] == 1
    assert stats["action_breakdown"]["AUTO_DECLINE"] == 1
    assert stats["feedback_submitted"] == 1
    assert stats["fraud_confirmation_rate"] == 1.0
    assert 0.0 < stats["avg_risk_score"] < 1.0


def test_stats_fraud_rate_all_legitimate(store):
    _decision(store, "aud_leg01", "TXN_L1", "ALLOW", 0.05)
    _decision(store, "aud_leg02", "TXN_L2", "ALLOW", 0.08)
    store.log_feedback("aud_leg01", "legitimate_confirmed")
    store.log_feedback("aud_leg02", "legitimate_confirmed")

    stats = store.get_aggregate_stats()
    assert stats["fraud_confirmation_rate"] == 0.0
