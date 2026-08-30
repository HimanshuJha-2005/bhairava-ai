"""
Bhairava — Fraud Detection System
app/services/audit_store.py

SQLite-backed persistent audit store for all fraud decisions.
Every auto-respond decision is written here for full traceability,
merchant feedback ingestion, and aggregate operational reporting.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DB_PATH = Path("data/audit/bhairava_audit.db")


class AuditStore:
    """
    Thread-safe SQLite store for fraud decision audit logs.

    Tables
    ------
    decisions  — one row per auto-respond evaluation, keyed by audit_id
    feedback   — merchant-submitted outcome labels, references decisions
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection (avoids cross-thread sharing)."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=10
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                audit_id        TEXT PRIMARY KEY,
                transaction_id  TEXT NOT NULL,
                action          TEXT NOT NULL,
                risk_score      REAL NOT NULL,
                risk_tier       TEXT NOT NULL,
                confidence      REAL NOT NULL,
                reasons         TEXT NOT NULL,
                requires_otp    INTEGER NOT NULL,
                decided_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_txn_id
                ON decisions (transaction_id);

            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id        TEXT NOT NULL REFERENCES decisions(audit_id),
                outcome         TEXT NOT NULL,
                submitted_at    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_audit_id
                ON feedback (audit_id);
            """
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_decision(
        self,
        audit_id: str,
        transaction_id: str,
        action: str,
        risk_score: float,
        risk_tier: str,
        confidence: float,
        reasons: list,
        requires_otp: bool,
        decided_at: datetime,
    ) -> None:
        """
        Persist a single fraud-decisioning event.
        Uses INSERT OR IGNORE to make repeated calls idempotent.
        """
        conn = self._conn()
        conn.execute(
            """
            INSERT OR IGNORE INTO decisions
                (audit_id, transaction_id, action, risk_score, risk_tier,
                 confidence, reasons, requires_otp, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                transaction_id,
                action,
                risk_score,
                risk_tier,
                confidence,
                json.dumps(reasons),
                int(requires_otp),
                decided_at.isoformat(),
            ),
        )
        conn.commit()

    def log_feedback(self, audit_id: str, outcome: str) -> bool:
        """
        Record a merchant-submitted outcome label for an existing decision.
        Returns False if the audit_id does not exist in the decisions table.
        """
        conn = self._conn()
        exists = conn.execute(
            "SELECT 1 FROM decisions WHERE audit_id = ?", (audit_id,)
        ).fetchone()
        if not exists:
            return False

        conn.execute(
            "INSERT INTO feedback (audit_id, outcome, submitted_at) VALUES (?, ?, ?)",
            (audit_id, outcome, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_decision_history(self, transaction_id: str) -> list:
        """
        Return all decision records for a given transaction_id,
        joined with any merchant-submitted feedback, ordered latest-first.
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT
                d.audit_id,
                d.transaction_id,
                d.action,
                d.risk_score,
                d.risk_tier,
                d.confidence,
                d.reasons,
                d.requires_otp,
                d.decided_at,
                f.outcome,
                f.submitted_at AS feedback_at
            FROM decisions d
            LEFT JOIN feedback f ON d.audit_id = f.audit_id
            WHERE d.transaction_id = ?
            ORDER BY d.decided_at DESC
            """,
            (transaction_id,),
        ).fetchall()

        result = []
        for r in rows:
            entry = dict(r)
            entry["reasons"] = json.loads(entry["reasons"])
            entry["requires_otp"] = bool(entry["requires_otp"])
            result.append(entry)
        return result

    def get_aggregate_stats(self) -> dict:
        """
        Compute live operational metrics across all persisted decisions.
        Returns a serialisable dict suitable for the /stats API response.
        """
        conn = self._conn()

        total = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        if total == 0:
            return {
                "total_evaluated": 0,
                "action_breakdown": {
                    "ALLOW": 0,
                    "CHALLENGE_3DS": 0,
                    "AUTO_DECLINE": 0,
                },
                "avg_risk_score": 0.0,
                "fraud_confirmation_rate": None,
                "feedback_submitted": 0,
            }

        action_rows = conn.execute(
            "SELECT action, COUNT(*) AS cnt FROM decisions GROUP BY action"
        ).fetchall()
        action_breakdown = {r["action"]: r["cnt"] for r in action_rows}
        for key in ("ALLOW", "CHALLENGE_3DS", "AUTO_DECLINE"):
            action_breakdown.setdefault(key, 0)

        avg_risk = conn.execute(
            "SELECT AVG(risk_score) FROM decisions"
        ).fetchone()[0]

        feedback_total = conn.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

        fraud_confirmed = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE outcome = 'fraud_confirmed'"
        ).fetchone()[0]

        fraud_rate: Optional[float] = (
            round(fraud_confirmed / feedback_total, 4) if feedback_total > 0 else None
        )

        return {
            "total_evaluated": total,
            "action_breakdown": action_breakdown,
            "avg_risk_score": round(avg_risk, 4),
            "fraud_confirmation_rate": fraud_rate,
            "feedback_submitted": feedback_total,
        }


# ---------------------------------------------------------------------------
# Global singleton — imported by auto_responder and endpoints
# ---------------------------------------------------------------------------
audit_store = AuditStore()
