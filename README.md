# Bhairava AI

### Fraud Detection + Automated Response System

Bhairava AI is an intelligent fraud prevention system designed to detect suspicious payment transactions, assess their risk, and automatically determine an appropriate response.

Built for the **Razorpay AI Buildathon 2026**.

---

## Project Status

🚧 **Currently in development** — ML Pipeline Complete | Building Response Layer

### Completed
- [x] Preprocessing & memory optimization
- [x] Domain-driven feature engineering
- [x] Chronological data splitting
- [x] Logistic Regression baseline
- [x] Imbalance-aware XGBoost training
- [x] Validation threshold optimization
- [x] Final untouched test evaluation

### Still to Build
- [ ] Risk scoring layer
- [ ] Automated response engine
- [ ] Decision explainability
- [ ] API / Application layer
- [ ] Monitoring and testing frameworks
- [ ] Final end-to-end system validation

---

## Dataset

Bhairava AI currently uses the **IEEE-CIS Fraud Detection** dataset.

The training dataset contains **590,540 transactions**:

| Metric | Value |
|---|---:|
| Total transactions | 590,540 |
| Legitimate transactions | 569,877 |
| Fraudulent transactions | 20,663 |
| Fraud rate | 3.50% |

The dataset presents a significant class imbalance, making metrics such as **precision, recall, F1-score, and PR-AUC** more informative than accuracy alone.

> Raw datasets are intentionally excluded from the repository because of their large size.

---

## Project Structure

```text
bhairava-ai/

│
├── app/                    # Application and API layer
├── data/                   # Local dataset storage
├── docs/                   # Project documentation
├── experiments/            # Data exploration and experiments
├── ml/                     # Machine learning pipeline
│   ├── data/               # Data preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training
│   └── evaluation/         # Model evaluation
├── tests/                  # Automated tests
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Current Findings

Initial analysis of the IEEE-CIS Fraud Detection dataset revealed a significant class imbalance.

Only **3.50%** of transactions in the training dataset are fraudulent:

- **Legitimate:** 569,877
- **Fraudulent:** 20,663
- **Fraud rate:** 3.50%

This makes fraud detection fundamentally different from a standard classification problem. A model that simply maximizes accuracy could appear highly effective while still failing to detect a meaningful number of fraudulent transactions.

Therefore, **precision, recall, F1-score, ROC-AUC, and PR-AUC** will be considered when evaluating the fraud detection models.

### Dataset Quality

The initial missing-value analysis also revealed substantial sparsity across several features:

- **174 features** contain more than 50% missing values.
- **2 features** contain more than 90% missing values.

This will be considered during feature selection and preprocessing rather than blindly imputing every missing value.

---

## Development Roadmap

### ✅ Phase 1: ML Pipeline (Completed)
- Preprocessing: Merged transaction and identity data, handled missing values, optimized memory (~2.5 GB to ~1.6 GB), encoded 31 categorical columns.
- Feature Engineering: Chronologically generated 33 domain-driven features (time, velocity, amount anomalies, identity signals).
- Splitting: Implemented strict chronological splits to prevent temporal leakage.
- Modeling: Trained a Logistic Regression baseline and an imbalance-aware XGBoost model.
- Optimization & Evaluation: Separated threshold optimization (on validation data) from final test evaluation to ensure real-world generalization.

### 🚧 Phase 2: Risk & Application Layer (Still to Build)
- Convert model probabilities into actionable risk scores.
- Design an automated response engine based on risk thresholds.
- Build decision explainability for flagged transactions.
- Develop the FastAPI application layer and endpoints.
- Implement system monitoring, testing, and final end-to-end validation.

---

## Tech Stack

- **Python** — core development language
- **Pandas** — data processing and analysis
- **NumPy** — numerical computation
- **Matplotlib** — data visualization
- **Scikit-learn** — machine learning and evaluation
- **XGBoost** — primary fraud detection modeling
- **Jupyter Notebook** — experimentation and analysis
- **FastAPI** — planned API layer

---

## ML Methodology & Model Results

Bhairava AI's core fraud detection pipeline is complete, designed heavily around rigorous evaluation and resistance to temporal leakage. The engineering methodology follows a strict sequence:

**DATA → PREPROCESSING → DOMAIN-DRIVEN FEATURE ENGINEERING → CHRONOLOGICAL SPLIT → LOGISTIC BASELINE → IMBALANCE-AWARE XGBOOST → VALIDATION THRESHOLD OPTIMIZATION → LOCKED THRESHOLD → UNTOUCHED TEST EVALUATION**

### 1. Feature Engineering & Chronological Splitting
To capture complex fraud behaviors without introducing temporal data leakage, **33 domain-driven features** were engineered chronologically. These include historical velocity signals, amount anomalies, and identity/email interactions.
- **Dimensionality:** Expanded the dataset from 435 merged columns to **468 columns**.
- **Chronological Split:** The 590,540 transactions were split strictly by time to simulate a real-world production environment where future data is unseen:
  - **Train:** 413,378 samples (3.52% fraud rate)
  - **Validation:** 59,053 samples (3.49% fraud rate)
  - **Test:** 118,109 samples (3.44% fraud rate)

### 2. Model Training & Baseline Comparison
Due to the dataset's extreme class imbalance, evaluating by standard accuracy is highly misleading. Models were evaluated prioritizing **PR-AUC and F1-score**. An imbalance-aware XGBoost model (`scale_pos_weight = 10.00`, best iteration: 923) was trained against a Logistic Regression baseline.

| Metric | Logistic Regression (Baseline) | XGBoost (Primary Model) |
|---|---|---|
| **Precision** | 0.1060 | 0.5375 |
| **Recall** | 0.7212 | 0.4606 |
| **F1-score** | 0.1849 | 0.4961 |
| **ROC-AUC** | 0.8242 | 0.8926 |
| **PR-AUC** | 0.1840 | **0.5077** |

### 3. Threshold Optimization & Final Test Evaluation
A critical design choice was separating threshold selection from final evaluation. Decision thresholds were optimized **strictly on the validation set** to prevent overfitting the test data. 

A locked threshold of **0.54** was selected based on validation performance (Validation F1: 0.5942, Precision: 0.6687, Recall: 0.5347, flagging 1,648 transactions).

Applying this locked threshold to the **untouched test set (118,109 samples)** yielded the final system performance:
- **Test Precision:** 0.5777
- **Test Recall:** 0.4437
- **Test F1-score:** 0.5019
- **Test ROC-AUC:** 0.8926
- **Test PR-AUC:** 0.5077

While these metrics highlight the sheer difficulty of identifying rare fraud in highly imbalanced data, they represent a mathematically robust, non-overfit baseline capable of generalizing to future unseen transactions. The project now moves into building the risk scoring and automated response layers on top of these predictions.

---

## Data & Reproducibility

The raw IEEE-CIS dataset is **not committed to this repository** because of its large size.

To reproduce the experiments:

1. Obtain the IEEE-CIS Fraud Detection dataset from its official source.
2. Place the required CSV files under:

```text
data/raw/ieee-fraud-detection/
```

3. Run the notebooks and scripts provided in the repository.

The expected dataset files are:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

---

## Disclaimer

Bhairava AI is an experimental fraud detection project developed for learning, research, and demonstration purposes.

It is **not intended to make autonomous financial decisions in production** without appropriate validation, monitoring, security controls, and human oversight.
