# Bhairava AI

### Fraud Detection + Automated Response System

Bhairava AI is an intelligent fraud prevention system designed to detect suspicious payment transactions, assess their risk, and automatically determine an appropriate response.

Built for the **Razorpay AI Buildathon 2026**.

---

## Project Status

🚧 **Currently in development**

### Completed

- [x] Feature engineering
- [x] Fraud detection preprocessing
- [x] Transaction and identity data integration
- [x] Memory-efficient preprocessing
- [x] Categorical encoding

### In Progress

- [ ] Fraud detection model
- [ ] Model evaluation
- [ ] Risk scoring
- [ ] Automated response engine
- [ ] API integration

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

### 1. Data Understanding

- [x] Dataset ingestion
- [x] Dataset profiling
- [x] Class distribution analysis
- [x] Transaction amount analysis
- [x] Missing-value analysis
- [ ] Feature relationship analysis

### 2. Data Preprocessing

- [x] Merge transaction and identity data
- [x] Handle missing values
- [x] Memory optimization
- [x] Encode categorical variables
- [ ] Identify and remove unsuitable features
- [ ] Prepare reproducible train/validation splits

### 3. Fraud Detection Models

- [ ] Establish a baseline model
- [ ] Train tree-based models
- [ ] Experiment with imbalance-aware learning
- [ ] Compare model performance
- [ ] Tune the strongest candidates

### 4. Evaluation

- [ ] Precision
- [ ] Recall
- [ ] F1-score
- [ ] ROC-AUC
- [ ] PR-AUC
- [ ] Confusion matrix
- [ ] Threshold analysis

### 5. Risk & Response Layer

- [ ] Convert model predictions into transaction risk scores
- [ ] Define risk-based decision thresholds
- [ ] Design automated response logic
- [ ] Document the reasoning behind each response level

### 6. Application Layer

- [ ] Build the fraud detection service
- [ ] Integrate the trained model
- [ ] Implement transaction scoring
- [ ] Add API endpoints
- [ ] Add logging and monitoring

### 7. Final Validation

- [ ] Evaluate the complete pipeline
- [ ] Test edge cases
- [ ] Verify reproducibility
- [ ] Document limitations
- [ ] Prepare final demonstration

---

## Tech Stack

- **Python** — core development language
- **Pandas** — data processing and analysis
- **NumPy** — numerical computation
- **Matplotlib** — data visualization
- **Scikit-learn** — machine learning and evaluation
- **Jupyter Notebook** — experimentation and analysis
- **FastAPI** — planned API layer

---

## Current Status

### Phase 2 — Feature Engineering & Preprocessing

Bhairava has now moved beyond initial dataset exploration and into the machine-learning pipeline.

The preprocessing and feature-engineering pipeline now:


- Merges transaction and identity data
- Handles missing values
- Optimizes memory usage
- Encodes categorical variables
- Generates fraud-related behavioral features

The feature-engineering pipeline currently adds **15 engineered features**, increasing the merged dataset from **435 columns to 450 columns** across **590,540 transactions**.

The next major milestone is **baseline fraud model development and evaluation**.

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