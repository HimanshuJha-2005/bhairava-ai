"""
Bhairava — Fraud Detection System
app/services/fraud_detector.py

Real-time model inference engine. Aligns incoming transaction payloads
with model feature specifications and executes sub-15ms risk scoring.
"""

import time
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from app.schemas.transaction import (
    TransactionPayload,
    FraudPredictionResponse,
    RiskTier,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"


class FraudDetectorService:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.locked_threshold = 0.38
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = MODEL_DIR / "bhairava_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run training pipeline first.")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            raise ValueError("Model missing feature_names_in_ metadata.")

        threshold_path = MODEL_DIR / "bhairava_threshold.json"
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                t_data = json.load(f)
                self.locked_threshold = float(t_data.get("threshold", 0.38))

    def _transform_payload_to_feature_row(self, payload: TransactionPayload) -> pd.DataFrame:
        """
        Extracts and aligns incoming payload attributes to model feature matrix.
        """
        raw = payload.model_dump()
        extra = raw.get("extra_features") or {}

        amt = float(raw["amount"])
        dt = raw.get("timestamp")

        hour = dt.hour if dt else 12
        day_of_week = dt.weekday() if dt else 2

        # Compute point-in-time engineered signals
        features: Dict[str, Any] = {
            "TransactionAmt": amt,
            "card1": float(raw.get("card1", 0)),
            "card2": float(raw.get("card2") or 300.0),
            "card3": float(raw.get("card3") or 150.0),
            "card5": float(raw.get("card5") or 226.0),
            "addr1": float(raw.get("addr1") or 299.0),
            "addr2": float(raw.get("addr2") or 87.0),
            "has_identity_data": int(raw.get("has_identity_data", 1)),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_night": 1 if hour < 6 else 0,
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "amount_zscore": (amt - 135.0) / 239.0,
            "amount_log": float(np.log1p(amt)),
            "amount_decimal": float(amt - np.floor(amt)),
            "is_round_amount": 1 if np.isclose(amt % 1, 0) else 0,
            "amount_vs_card_mean": extra.get("amount_vs_card_mean", 1.0),
            "amount_vs_card_addr_mean": extra.get("amount_vs_card_addr_mean", 1.0),
            "card1_txn_count": extra.get("card1_txn_count", 5),
            "card_addr_txn_count": extra.get("card_addr_txn_count", 3),
            "card_full_txn_count": extra.get("card_full_txn_count", 2),
            "email_txn_count": extra.get("email_txn_count", 100),
            "card_time_since_prev": extra.get("card_time_since_prev", 86400.0),
            "card_addr_time_since_prev": extra.get("card_addr_time_since_prev", 86400.0),
            "rapid_card_activity": 1 if extra.get("card_time_since_prev", 86400.0) < 3600 else 0,
            "unique_cards_per_addr": extra.get("unique_cards_per_addr", 1),
            "is_new_card_addr": 1 if extra.get("card_addr_txn_count", 3) == 0 else 0,
            "card_reuse_signal": float(np.log1p(extra.get("card1_txn_count", 5))),
            "email_domain_freq": extra.get("email_domain_freq", 500),
            "is_rare_email_domain": 1 if extra.get("email_domain_freq", 500) < 100 else 0,
            "email_domain_match": 1 if (raw.get("P_emaildomain") and raw.get("P_emaildomain") == raw.get("R_emaildomain")) else 0,
            "D1_anchor_day": extra.get("D1_anchor_day", 0.0),
            "D2_anchor_day": extra.get("D2_anchor_day", 0.0),
        }

        # Populate any provided V/C/D columns or fill unprovided with neutral defaults
        for col in self.feature_names:
            if col not in features:
                if col in extra:
                    features[col] = float(extra[col])
                elif col.startswith("V"):
                    features[col] = 1.0
                elif col.startswith("C"):
                    features[col] = 1.0
                elif col.startswith("D"):
                    features[col] = 0.0
                else:
                    features[col] = 0.0

        df_row = pd.DataFrame([features])[self.feature_names]
        return df_row

    def predict_risk(self, payload: TransactionPayload) -> Tuple[FraudPredictionResponse, dict]:
        """
        Executes model prediction and returns risk score + raw feature map.
        """
        start_time = time.perf_counter()

        df_row = self._transform_payload_to_feature_row(payload)
        proba = float(self.model.predict_proba(df_row)[0, 1])

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if proba < 0.35:
            tier = RiskTier.LOW
        elif proba < 0.65:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.HIGH

        confidence = float(abs(proba - 0.5) * 2.0)

        response = FraudPredictionResponse(
            transaction_id=payload.transaction_id,
            risk_score=round(proba, 4),
            risk_tier=tier,
            confidence=round(confidence, 4),
            inference_time_ms=round(latency_ms, 2),
        )

        raw_features = df_row.iloc[0].to_dict()
        return response, raw_features


# Global singleton
detector_service = FraudDetectorService()
