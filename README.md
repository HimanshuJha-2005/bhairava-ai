# Bhairava

Payment fraud risk scoring with a policy layer. A transaction is scored, mapped to an action, explained with TreeSHAP, and written to an audit ledger that accepts chargeback feedback.

Built for the Razorpay AI Buildathon 2026 (AI Risk Manager track).

## Results

Trained and evaluated on the IEEE-CIS Fraud Detection dataset: 590,540 transactions, 3.5% fraud. The split is chronological, not random:

| Split | Transactions | Fraud rate |
|---|---|---|
| Train (first 70%) | 413,378 | 3.52% |
| Validation (middle 10%) | 59,053 | 3.49% |
| Test (final 20%) | 118,109 | 3.44% |

The decision threshold (0.38) was chosen on validation data and then applied unchanged to the test set.

| Model | Precision | Recall | Fraud F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic regression baseline | 0.0875 | 0.7817 | 0.1574 | 0.8248 | 0.1855 |
| XGBoost | 0.5804 | 0.4877 | 0.5300 | 0.9056 | 0.5458 |

Accuracy is not reported as a headline number: predicting "legitimate" for every transaction scores 96.5% and catches no fraud. PR-AUC and fraud-class F1 are the metrics that matter here.

## Pipeline

1. **Features.** 457 columns from transaction and identity data, including UTC time features, amount fractions, card and address velocity, and card inception anchors. Historical aggregates are computed in chronological order so a transaction never sees later data.
2. **Scoring.** Regularized XGBoost (`colsample_bytree=0.7`, `subsample=0.8`, `min_child_weight=5`, `learning_rate=0.03`) returns a fraud probability.
3. **Policy.** Probability bands map to actions:

| Band | Action | Meaning |
|---|---|---|
| `< 0.35` | `ALLOW` | Continue without friction |
| `0.35 - 0.65` | `CHALLENGE_3DS` | Step-up verification |
| `> 0.65` | `AUTO_DECLINE` | Block and notify |

4. **Explanation.** TreeSHAP produces the top five feature contributions per transaction, each marked as increasing or reducing risk.
5. **Audit.** Decisions are stored in SQLite by audit ID and transaction ID. `POST /api/v1/feedback` records chargeback outcomes and `GET /api/v1/stats` returns aggregate counts.

A Streamlit dashboard provides a manual scoring view and displays the same explanation and ledger data.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/predict-fraud` | Score a transaction |
| POST | `/api/v1/auto-respond` | Score, decide, explain, and store |
| GET | `/api/v1/health` | Service and model status |
| POST | `/api/v1/feedback` | Record a confirmed outcome |
| GET | `/api/v1/audit/{transaction_id}` | Decision history for a transaction |
| GET | `/api/v1/stats` | Aggregate ledger statistics |

Request bodies use the IEEE-CIS field names where applicable (`card1`, `card2`, `card3`, `addr1`, `P_emaildomain`, and so on) plus `transaction_id` and `amount`.

## Stack

- Python, FastAPI, Pydantic
- XGBoost, scikit-learn, pandas, NumPy
- TreeSHAP through the XGBoost explainer
- SQLite for the audit ledger
- Streamlit and Plotly for the dashboard

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload    # API on :8000, docs at /docs
streamlit run dashboard/main.py  # dashboard on :8501
python -m pytest tests/ -v       # 15 tests
```

The service expects trained artifacts at `data/models/bhairava_model.pkl` and `data/models/bhairava_threshold.json`. These are not committed because the IEEE-CIS dataset is not redistributable. Without them the API exits at startup with a clear error, and the test suite cannot import the app. The evaluation code and reported metrics are included in `ml/`.

## Limitations

- Recall is 0.49 on the test set: roughly half of fraud is missed at this threshold. The confusion matrix is deliberately reported rather than hidden behind accuracy.
- The model is trained on 2018 card-not-present data. Fraud patterns drift, so production use would require periodic retraining on recent chargebacks.
- The audit ledger is local SQLite, not a multi-instance store.
- No authentication on the API; it is a benchmark service, not a production gateway.

## Layout

```
app/                 FastAPI app, schemas, services, endpoints
dashboard/           Streamlit operations view
ml/                  feature engineering, training, evaluation, SHAP
tests/               API, policy engine, ledger, and SHAP tests
demo.py              CLI walkthrough
```
