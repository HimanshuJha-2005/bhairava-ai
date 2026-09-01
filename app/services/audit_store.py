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

    def get_recent_decisions(self, limit: int = 20) -> list:
        """
        Return the N most recent decisions across all transactions,
        joined with any merchant-submitted feedback, ordered latest-first.
        Used by the monitoring dashboard.
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
            ORDER BY d.decided_at DESC
            LIMIT ?
            """,
            (limit,),
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

    def reset_store(self) -> None:
        """Clear all decisions and feedback records."""
        conn = self._conn()
        conn.execute("DELETE FROM feedback")
        conn.execute("DELETE FROM decisions")
        conn.commit()

    def seed_realistic_gateway_traffic(self, total_samples: int = 150) -> None:
        """
        Populate the audit store with a realistic payment gateway distribution:
        - ~92% ALLOW (Low risk legitimate transactions, avg risk ~0.03)
        - ~5.5% CHALLENGE 3DS (Medium risk transactions, avg risk ~0.45)
        - ~2.5% AUTO DECLINE (High risk blocked fraud, avg risk ~0.82)
        """
        import random
        from datetime import timedelta

        self.reset_store()
        conn = self._conn()
        now = datetime.now(timezone.utc)

        # Generate realistic transactions
        for i in range(total_samples):
            audit_id = f"aud_rzp_{i+1000:04d}"
            txn_id = f"pay_live_{random.randint(100000, 999999)}"
            offset_seconds = random.randint(10, 86400)
            decided_at = (now - timedelta(seconds=offset_seconds)).isoformat()

            rand_val = random.random()
            if rand_val < 0.92:
                # Legitimate transaction
                action = "ALLOW"
                risk_score = round(random.uniform(0.01, 0.18), 4)
                risk_tier = "LOW"
                confidence = round((1.0 - risk_score), 4)
                reasons = json.dumps(["LOW_RISK_NORMAL_BEHAVIOR"])
                requires_otp = 0
            elif rand_val < 0.975:
                # Step-up 3DS challenge
                action = "CHALLENGE_3DS"
                risk_score = round(random.uniform(0.36, 0.58), 4)
                risk_tier = "MEDIUM"
                confidence = round(abs(risk_score - 0.5) * 2, 4)
                reasons = json.dumps(["FIRST_OBSERVED_CARD_OR_ADDRESS_COMBINATION", "TRANSACTION_AMOUNT_UNUSUALLY_HIGH_FOR_CARD"])
                requires_otp = 1
            else:
                # High risk decline
                action = "AUTO_DECLINE"
                risk_score = round(random.uniform(0.68, 0.94), 4)
                risk_tier = "HIGH"
                confidence = round(risk_score, 4)
                reasons = json.dumps(["HIGH_FREQUENCY_RAPID_CARD_REUSE_DETECTED", "PURCHASER_AND_RECIPIENT_EMAIL_DOMAIN_MISMATCH"])
                requires_otp = 0

            conn.execute(
                """
                INSERT INTO decisions
                    (audit_id, transaction_id, action, risk_score, risk_tier,
                     confidence, reasons, requires_otp, decided_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (audit_id, txn_id, action, risk_score, risk_tier, confidence, reasons, requires_otp, decided_at),
            )

        # Seed sample feedback
        conn.execute("INSERT INTO feedback (audit_id, outcome, submitted_at) VALUES ('aud_rzp_1001', 'legitimate_confirmed', ?)", (now.isoformat(),))
        conn.execute("INSERT INTO feedback (audit_id, outcome, submitted_at) VALUES ('aud_rzp_1002', 'legitimate_confirmed', ?)", (now.isoformat(),))
        conn.execute("INSERT INTO feedback (audit_id, outcome, submitted_at) VALUES ('aud_rzp_1145', 'fraud_confirmed', ?)", (now.isoformat(),))
        conn.commit()


# ---------------------------------------------------------------------------
# Global singleton — imported by auto_responder and endpoints
# ---------------------------------------------------------------------------
audit_store = AuditStore()
