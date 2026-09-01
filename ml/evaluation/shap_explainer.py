"""
Bhairava — Fraud Detection System
ml/evaluation/shap_explainer.py

TreeSHAP-powered local explainability layer.
Computes mathematically rigorous Shapley attribution values per transaction,
identifying the exact top contributing features driving each fraud decision.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
from app.services.fraud_detector import detector_service


# Human-friendly descriptions for common IEEE-CIS feature prefixes
FEATURE_FRIENDLY_NAMES = {
    "TransactionAmt": "Transaction Amount",
    "amount_decimal": "Cents / Decimal Fraction Pattern",
    "card1": "Bank Identification Hash (BIN)",
    "card2": "Card Category Code",
    "card3": "Card Country Code",
    "card_full_txn_count": "Composite Card Reuse Velocity",
    "email_domain_match": "Purchaser vs Recipient Domain Match",
    "P_emaildomain": "Purchaser Email Domain Risk",
    "R_emaildomain": "Recipient Email Domain Risk",
    "addr1": "Billing Region Zip Prefix",
    "addr2": "Billing Country",
    "D1_anchor_day": "Card Inception Anchor Days",
    "has_identity_data": "Device Telemetry Presence",
    "V258": "Identity Verification Consistency Signal",
    "V257": "Device Authorization Rate Signal",
    "V70": "Card Present Transaction Velocity",
    "V294": "Historical Merchant Chargeback Count",
    "V201": "Cardholder Address Verification Match",
    "C4": "Historical High-Velocity Transaction Count",
    "C1": "Card Profile Transaction Count",
    "C13": "Cumulative Device Usage Count",
    "C14": "Transaction Authorization History Count",
    "id_30": "Operating System / Device Fingerprint",
    "id_31": "Browser Family Telemetry",
    "id_33": "Screen Resolution Telemetry",
    "DeviceInfo": "Hardware Device Profile",
}


class ShapExplainerService:
    def __init__(self):
        self._explainer: Optional[shap.TreeExplainer] = None
        self._expected_value: float = 0.0

    def _get_explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            if detector_service.model is None:
                raise RuntimeError("Fraud detector XGBoost model not loaded.")
            # Use TreeExplainer for fast exact tree-based Shapley computation
            self._explainer = shap.TreeExplainer(detector_service.model)
            expected_val = self._explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                self._expected_value = float(expected_val[-1])
            else:
                self._expected_value = float(expected_val)
        return self._explainer

    def explain_transaction(
        self,
        feature_map: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Calculates exact TreeSHAP values for a single transaction vector.
        Returns top-k features sorted by absolute Shapley contribution.
        """
        explainer = self._get_explainer()
        feature_names = detector_service.feature_names

        # Construct single-row DataFrame aligned to exact model feature columns
        row_dict = {col: feature_map.get(col, 0.0) for col in feature_names}
        df_row = pd.DataFrame([row_dict], columns=feature_names)

        # Compute SHAP values for the single sample
        shap_values = explainer.shap_values(df_row)
        if isinstance(shap_values, list):
            sample_shap = shap_values[-1][0]
        elif len(shap_values.shape) == 2:
            sample_shap = shap_values[0]
        else:
            sample_shap = shap_values

        # Pair features with their SHAP values
        contributions = []
        for feat_name, s_val in zip(feature_names, sample_shap):
            s_float = float(s_val)
            name_str = str(feat_name)
            if abs(s_float) > 1e-5:
                contributions.append({
                    "feature": name_str,
                    "friendly_name": FEATURE_FRIENDLY_NAMES.get(name_str, name_str),
                    "shap_value": round(s_float, 4),
                    "direction": "increases_risk" if s_float > 0 else "reduces_risk",
                    "abs_impact": abs(s_float),
                })

        # Sort by absolute impact
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_features = contributions[:top_k]

        # Clean output dict
        cleaned_top = [
            {
                "feature": str(f["feature"]),
                "friendly_name": str(f["friendly_name"]),
                "shap_value": float(f["shap_value"]),
                "direction": str(f["direction"]),
            }
            for f in top_features
        ]

        return {
            "base_score": round(self._expected_value, 4),
            "top_features": cleaned_top,
        }


# Global singleton
shap_service = ShapExplainerService()
