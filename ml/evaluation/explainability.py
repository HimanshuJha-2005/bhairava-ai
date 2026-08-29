"""
Bhairava — Fraud Detection System
ml/evaluation/explainability.py

Extracts model feature importances and maps individual transaction
signals to explainable, human-readable reason codes for merchants.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"


def export_feature_importance():
    """
    Extracts and saves feature importances from the trained Bhairava model.
    """
    model_path = MODEL_DIR / "bhairava_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature_importances_")
        return {}

    feature_names = getattr(model, "feature_names_in_", None)
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_dict = {
        name: float(imp)
        for name, imp in zip(feature_names, importances)
    }

    # Sort descending
    sorted_importances = dict(
        sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    )

    output_path = MODEL_DIR / "bhairava_feature_importance.json"
    with open(output_path, "w") as f:
        json.dump(sorted_importances, f, indent=4)

    print(f"Feature importance saved to: {output_path}")
    return sorted_importances


def explain_transaction(features: Dict[str, Any], risk_score: float) -> List[str]:
    """
    Rule-based attribution layer translating raw transaction features
    and anomalies into merchant-friendly risk reason codes.
    """
    reasons = []

    # Amount anomalies
    amt = float(features.get("TransactionAmt", 0.0))
    amt_vs_card = float(features.get("amount_vs_card_mean", 1.0))
    amt_zscore = float(features.get("amount_zscore", 0.0))
    amt_decimal = float(features.get("amount_decimal", 0.0))

    if amt_vs_card > 3.0:
        reasons.append("TRANSACTION_AMOUNT_UNUSUALLY_HIGH_FOR_CARD")
    elif amt_zscore > 4.0:
        reasons.append("HIGH_GLOBAL_AMOUNT_ANOMALY")

    if amt_decimal > 0 and abs(amt_decimal - round(amt_decimal, 2)) > 1e-4:
        reasons.append("IRREGULAR_CURRENCY_FRACTIONAL_DECIMAL")

    # Velocity and card history
    card_txns = int(features.get("card1_txn_count", 5))
    rapid_activity = int(features.get("rapid_card_activity", 0))
    time_since_prev = float(features.get("card_time_since_prev", -1))
    is_new_card_addr = int(features.get("is_new_card_addr", 0))

    if card_txns == 0 or is_new_card_addr == 1:
        reasons.append("FIRST_OBSERVED_CARD_OR_ADDRESS_COMBINATION")

    if rapid_activity == 1 or (0 <= time_since_prev < 600):
        reasons.append("HIGH_FREQUENCY_RAPID_CARD_REUSE_DETECTED")

    # Email mismatch or rare domain
    if int(features.get("email_domain_match", 1)) == 0:
        reasons.append("PURCHASER_AND_RECIPIENT_EMAIL_DOMAIN_MISMATCH")

    if int(features.get("is_rare_email_domain", 0)) == 1:
        reasons.append("UNCOMMONLY_OBSERVED_EMAIL_DOMAIN")

    # Identity data
    if int(features.get("has_identity_data", 1)) == 0:
        reasons.append("MISSING_DEVICE_OR_IDENTITY_TELEMETRY")

    # Off hours
    if int(features.get("is_night", 0)) == 1 and risk_score >= 0.30:
        reasons.append("OFF_HOURS_HIGH_RISK_TRANSACTION")

    # Return clean status if no anomalies and score is low
    if not reasons:
        if risk_score < 0.20:
            return ["LOW_RISK_NORMAL_BEHAVIOR"]
        else:
            return ["ANOMALOUS_BEHAVIORAL_PATTERN_DETECTED"]

    return reasons


if __name__ == "__main__":
    importances = export_feature_importance()
    print("\nTop 10 Most Predictive Features:")
    for rank, (feat, imp) in enumerate(list(importances.items())[:10], 1):
        print(f"  {rank:>2}. {feat:<30} {imp * 100:.2f}%")
