<div align="center">

# ⚡ Bhairava AI
### Real-Time Payment Risk Manager & Policy Auto-Responder Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed%20(100%25)-success?style=for-the-badge)](https://pytest.org)

**Built for the Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**

*Pairing imbalance-aware gradient boosting with cost-optimal 3-tier policy automation, TreeSHAP mathematical attribution, and persistent audit ledgers.*

</div>

---

## 🎬 Demo

[![Bhairava AI Demo](https://img.youtube.com/vi/kVo1YVVl3DM/maxresdefault.jpg)](https://youtu.be/kVo1YVVl3DM)

> Click the thumbnail above to watch the full system walkthrough.

---

## 📌 Executive Summary

Traditional fraud detection forces merchants into an expensive dilemma: **overly strict rules cause high false positives that reject genuine buyers**, while **loose rules cause catastrophic chargeback liabilities**.

**Bhairava AI** is an intelligent, two-stage payment defense system designed for modern payment gateways and merchants:
1. **Stage 1 (ML Risk Engine):** Evaluates 457 continuous behavioral and telemetry features in **< 12ms** to output an un-overfit fraud risk probability (**0.9056 ROC-AUC**, **0.5458 PR-AUC** on 118k untouched test samples).
2. **Stage 2 (Policy Auto-Responder):** Converts raw risk into business-aware actions:
   - `ALLOW` (Zero-friction 1-click checkout for verified shoppers)
   - `CHALLENGE_3DS` (Triggers Step-Up OTP verification to save borderline sales)
   - `AUTO_DECLINE` (Instantly blocks high-risk attacks to eliminate chargebacks)
3. **Stage 3 (TreeSHAP Explainability):** Generates exact mathematical Shapley feature attributions per transaction in real-time.
4. **Stage 4 (Closed-Loop Audit Ledger):** Persists all decisions to SQLite with dispute feedback tracking (`POST /api/v1/feedback`) and live operational analytics (`GET /api/v1/stats`).

---

## 🏛️ End-to-End System Architecture

```
Incoming Payment Payload (₹ INR)
               │
               ▼
[ 1. Preprocessing & Leakage-Free Feature Pipeline ]
  • Chronological point-in-time state (Zero lookahead bias)
  • Currency decimal/cents fraction analysis
  • Composite card identity velocity (card1_card2_card3_card5_addr1)
  • Card inception anchor timeline (TransactionDT / 86400 - D1)
               │
               ▼
[ 2. Stage 1: Regularized XGBoost Risk Engine ]
  • Sub-12ms inference latency over 457 feature matrix
  • Test ROC-AUC: 0.9056 | PR-AUC: 0.5458 | F1: 0.5300
  • Validation-locked decision threshold: 0.38
               │
               ▼
[ 3. Stage 2: Policy Auto-Responder Engine ]
  ├─ Risk < 0.35  ────────► ACTION: ALLOW          (Bypass friction, instant checkout)
  ├─ 0.35 ≤ Risk ≤ 0.65 ──► ACTION: CHALLENGE_3DS  (Trigger Step-Up OTP challenge)
  └─ Risk > 0.65  ────────► ACTION: AUTO_DECLINE   (Instant block + merchant security alert)
               │
               ▼
[ 4. Stage 3: TreeSHAP Mathematical Attribution ]
  • Top-5 feature Shapley contributions per transaction
  • Explicit directionality: increases_risk vs reduces_risk
               │
               ▼
[ 5. Stage 4: Persistent Audit Ledger & Closed-Loop Feedback ]
  • Thread-safe SQLite store indexed by audit_id and transaction_id
  • Merchant chargeback feedback ingestion (/feedback)
  • Live aggregate telemetry reporting (/stats)
               │
               ▼
[ 6. Interfaces & Delivery ]
  • Production FastAPI REST Layer (< 15ms latency)
  • Live Fintech Operations Center (Streamlit UI)
```

---

## 📊 Dataset & The 3.5% Imbalance Reality

Bhairava AI is trained and benchmarked on the **IEEE-CIS Fraud Detection Benchmark** (**590,540 transactions**).

| Dataset Metric | Benchmark Value |
| :--- | :--- |
| **Total Transactions** | 590,540 |
| **Legitimate Transactions** | 569,877 (96.50%) |
| **Fraudulent Transactions** | 20,663 (3.50%) |
| **Merged Columns** | 435 raw attributes (Transactions + Identity) |
| **Engineered Feature Space** | 457 continuous dimensions |

### Why Raw Accuracy is a Dangerous Vanity Metric

In real payment traffic with a 3.5% fraud rate, a naive model that predicts "legitimate" on every transaction achieves **96.50% accuracy** while catching **0% of fraud**.

Bhairava AI is evaluated exclusively on **PR-AUC, ROC-AUC, Precision, Recall, and fraud-class F1-Score**.

---

## 🔬 ML Methodology & Benchmark Results

### 1. Strict Chronological Split (Zero Future Leakage)

To simulate real-world production where future transactions are unseen, data was partitioned strictly across the `TransactionDT` timeline:
- **Train Set (First ~70%):** 413,378 transactions (3.52% fraud rate)
- **Validation Set (Middle ~10%):** 59,053 transactions (3.49% fraud rate)
- **Untouched Test Set (Final ~20%):** 118,109 transactions (3.44% fraud rate)

### 2. Model Performance on Untouched Test Set (118,109 samples)

Decision thresholds were optimized strictly on validation data (**Validation-Locked Threshold = 0.38**), then evaluated on the untouched test set:

| Model Architecture | Precision | Recall | Fraud F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.0875 | **0.7817** | 0.1574 | 0.8248 | 0.1855 |
| **XGBoost (Bhairava AI)** | **0.5804** | 0.4877 | **0.5300** | **0.9056** | **0.5458** |

```text
Final Untouched Test Set Classification Report:
              precision    recall  f1-score   support

  Legitimate       0.98      0.99      0.98    114045
       Fraud       0.58      0.49      0.53      4064

    accuracy                           0.97    118109
   macro avg       0.78      0.74      0.76    118109
weighted avg       0.97      0.97      0.97    118109
```

---

## 🛠️ What Broke & How We Got Out

### Problem: The Feature Bloat Trap

Early in development, we engineered 6 manual composite boolean flags combining night timestamps, new card BINs, and large transaction amounts into single `is_high_risk_combo` boolean flags.

- **The Failure:** Fraud F1 dropped from 0.5067 to 0.5019 and precision degraded.
- **Root Cause:** Gradient boosted decision trees already learn non-linear threshold boundaries on continuous inputs. Forcing rigid boolean splits fragmented tree depth, created collinear noise across 450+ columns, and over-penalized borderline legitimate transactions.

### The Engineering Recovery

1. **Pruned Boolean Noise:** Removed all rigid boolean flags.
2. **High-Signal Continuous Features:** Added fractional cents distribution, composite identity velocity (`card1_card2_card3_card5_addr1`), and point-in-time inception anchors (`TransactionDT / 86400 - D1`).
3. **Subsampling Regularization:** Applied tree subsampling (`colsample_bytree=0.70`, `subsample=0.80`, `min_child_weight=5`, `learning_rate=0.03`) to prevent over-reliance on high-frequency card IDs.
4. **Validation-Locked Thresholding:** Decoupled threshold selection from test data, locking the decision boundary at **0.38** on validation set only.

**Result:** Test ROC-AUC broke 0.90 (reaching **0.9056**), PR-AUC jumped to **0.5458**, and test F1 reached **0.5300** on 118,109 untouched test transactions.

---

## 🛡️ Stage 2: Policy Auto-Responder Engine

| Risk Tier | Probability Range | Action | Business Rationale |
| :--- | :---: | :---: | :--- |
| **LOW** | `< 0.35` | `ALLOW` | Zero-friction checkout for verified customers. |
| **MEDIUM** | `0.35 – 0.65` | `CHALLENGE_3DS` | Step-Up OTP verification. Preserves legitimate sales while catching fraudsters who lack OTP access. |
| **HIGH** | `> 0.65` | `AUTO_DECLINE` | Instant block + critical merchant webhook alert to eliminate chargeback liability. |

---

## 🧠 Stage 3: TreeSHAP Mathematical Attribution

Every transaction evaluation returns exact **TreeSHAP** (Shapley Additive Explanations) feature contribution values:

```json
"shap_attribution": {
  "base_score": -0.9315,
  "top_features": [
    {
      "feature": "card_full_txn_count",
      "friendly_name": "Composite Card Reuse Velocity",
      "shap_value": 0.3120,
      "direction": "increases_risk"
    },
    {
      "feature": "TransactionAmt",
      "friendly_name": "Transaction Amount",
      "shap_value": 0.2870,
      "direction": "increases_risk"
    },
    {
      "feature": "D1_anchor_day",
      "friendly_name": "Card Inception Anchor Days",
      "shap_value": -0.1430,
      "direction": "reduces_risk"
    }
  ]
}
```

---

## 🚀 Quickstart & How to Run

### 1. Installation

```powershell
git clone https://github.com/HimanshuJha-2005/bhairava-ai.git
cd bhairava-ai
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Live Operations Center (Streamlit UI)

```powershell
streamlit run dashboard/main.py
```

> Opens at `http://localhost:8501`

### 3. Launch REST API Server

```powershell
uvicorn app.main:app --reload
```

- Swagger UI: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

### 4. Run Automated Test Suite

```powershell
python -m pytest tests/ -v
```

> 15 / 15 tests passing.

---

## 🔌 API Reference

### `POST /api/v1/auto-respond`

**Request:**
```json
{
  "transaction_id": "pay_rzp_live_9921",
  "amount": 2499.00,
  "card1": 13926,
  "card2": 321.0,
  "card3": 150.0,
  "addr1": 299.0,
  "P_emaildomain": "gmail.com",
  "has_identity_data": 1
}
```

**Response (200 OK):**
```json
{
  "transaction_id": "pay_rzp_live_9921",
  "action": "ALLOW",
  "risk_score": 0.0023,
  "risk_tier": "LOW",
  "confidence": 0.9954,
  "reasons": ["LOW_RISK_NORMAL_BEHAVIOR"],
  "shap_attribution": {
    "base_score": -0.9315,
    "top_features": [
      {
        "feature": "TransactionAmt",
        "friendly_name": "Transaction Amount",
        "shap_value": -0.4120,
        "direction": "reduces_risk"
      }
    ]
  },
  "requires_otp_challenge": false,
  "merchant_notification": {
    "enabled": false,
    "severity": "INFO",
    "title": "Transaction Cleared",
    "message": "Transaction pay_rzp_live_9921 verified as legitimate (Risk: 0.23%).",
    "action_required": "NONE"
  },
  "decision_timestamp": "2026-09-01T12:00:00Z",
  "audit_id": "aud_858b100b09f1"
}
```

### `POST /api/v1/feedback`
Submit merchant-confirmed dispute outcome for any `audit_id`.

### `GET /api/v1/stats`
Live aggregate operational metrics across the persistent SQLite audit ledger.

---

## 📁 Repository Structure

```text
bhairava-ai/
├── app/
│   ├── api/v1/endpoints.py        # FastAPI REST routes
│   ├── schemas/
│   │   ├── transaction.py         # Pydantic payloads & SHAP schemas
│   │   └── audit.py               # Feedback & stats schemas
│   ├── services/
│   │   ├── fraud_detector.py      # XGBoost inference engine
│   │   ├── auto_responder.py      # 3-tier policy decision engine
│   │   └── audit_store.py         # Thread-safe SQLite audit ledger
│   └── main.py                    # FastAPI application entrypoint
├── dashboard/
│   ├── main.py                    # Streamlit Operations Center
│   └── components.py              # Plotly charts & SHAP visualizations
├── data/
│   ├── audit/                     # SQLite audit database
│   └── models/                    # XGBoost model & threshold artifacts
├── ml/
│   ├── data/preprocessing.py      # Ingestion & memory reduction (36.2% saved)
│   ├── features/feature_engineering.py  # Leak-free point-in-time features
│   ├── models/training.py         # Regularized XGBoost training
│   └── evaluation/
│       ├── threshold.py           # Validation threshold optimization (0.38)
│       ├── final_evaluation.py    # Untouched test set evaluation
│       ├── explainability.py      # Feature importance & reason attribution
│       └── shap_explainer.py      # TreeSHAP mathematical explainability
├── tests/
│   ├── test_api.py                # REST API integration tests
│   ├── test_auto_responder.py     # Policy engine unit tests
│   ├── test_audit_store.py        # SQLite audit ledger tests
│   └── test_shap_explainer.py     # TreeSHAP unit tests
├── demo.py                        # Interactive CLI demo simulation
├── README.md
└── requirements.txt
```

---

## ⚠️ Limitations

- **Recall is 0.49** — Bhairava misses roughly half of actual fraud cases on the test set. This is an honest number, not a footnote.
- Trained on IEEE-CIS 2018 benchmark data. Fraud patterns evolve — production deployment requires quarterly retraining with live transaction feedback.
- The closed-loop audit ledger exists precisely for this: merchants confirm real fraud cases, enabling continuous model improvement over time.

---

## 👨‍💻 Author

Built by **Himanshu Jha** for the **Razorpay AI Buildathon 2026** *(Track 2: AI Risk Manager)*.
