"""
Bhairava — Fraud Detection System
ml/models/training.py

Trains and compares fraud detection models.

Pipeline:
1. Load + engineer features
2. Time-based train/validation/test split
3. Train Logistic Regression baseline
4. Train XGBoost
5. Compare models
6. Save the best model
7. Save validation + test probabilities
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.data.preprocessing import get_clean_data
from ml.features.feature_engineering import engineer_features


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

def get_features_and_target(df):
    """Separate model features from the target."""

    drop_cols = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
    ]

    feature_cols = [
        col for col in df.columns
        if col not in drop_cols
    ]

    X = df[feature_cols]
    y = df["isFraud"]

    return X, y


# ---------------------------------------------------------
# Time-based split
# ---------------------------------------------------------

def time_based_split(
    df,
    train_ratio=0.7,
    validation_ratio=0.1,
):
    """Create chronological train/validation/test splits."""

    df_sorted = (
        df.sort_values("TransactionDT")
        .reset_index(drop=True)
    )

    train_end = int(
        len(df_sorted) * train_ratio
    )

    validation_end = int(
        len(df_sorted)
        * (train_ratio + validation_ratio)
    )

    train = df_sorted.iloc[:train_end]
    validation = df_sorted.iloc[
        train_end:validation_end
    ]
    test = df_sorted.iloc[validation_end:]

    print("\nTime-based split:")

    print(
        f"  Train:      {len(train):,} rows | "
        f"Fraud rate: {train['isFraud'].mean() * 100:.2f}%"
    )

    print(
        f"  Validation: {len(validation):,} rows | "
        f"Fraud rate: {validation['isFraud'].mean() * 100:.2f}%"
    )

    print(
        f"  Test:       {len(test):,} rows | "
        f"Fraud rate: {test['isFraud'].mean() * 100:.2f}%"
    )

    return train, validation, test


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    """Evaluate a trained model."""

    y_pred = model.predict(X_test)

    y_proba = (
        model.predict_proba(X_test)[:, 1]
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    pr_auc = average_precision_score(
        y_test,
        y_proba,
    )

    print("\n" + "=" * 60)
    print(f"{model_name} Results")
    print("=" * 60)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")

    print("\nFull Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            zero_division=0,
        )
    )

    return {
        "model_name": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "model": model,
        "y_proba": y_proba,
    }


# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

def train_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Logistic Regression baseline.

    StandardScaler improves convergence because
    features have different numerical scales.
    """

    print(
        "\nTraining Logistic Regression baseline..."
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    return evaluate_model(
        model,
        X_test,
        y_test,
        "Logistic Regression (Baseline)",
    )


# ---------------------------------------------------------
# XGBoost
# ---------------------------------------------------------

def train_xgboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    Train XGBoost using validation data for early stopping.
    """

    print("\nTraining XGBoost...")

    neg = (
        y_train == 0
    ).sum()

    pos = (
        y_train == 1
    ).sum()

    original_weight = neg / pos

    # Cap class weighting to avoid excessive
    # pressure toward the minority class.
    scale_pos_weight = min(
        original_weight,
        10,
    )

    print(
        f"  Original class ratio: "
        f"{original_weight:.2f}"
    )

    print(
        f"  scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    model = XGBClassifier(
        n_estimators=2000,
        max_depth=7,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.70,
        subsample=0.80,
        min_child_weight=5,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=100,
    )

    print(
        f"  Best iteration: "
        f"{model.best_iteration}"
    )

    return evaluate_model(
        model,
        X_test,
        y_test,
        "XGBoost (Bhairava)",
    )


# ---------------------------------------------------------
# Compare + save
# ---------------------------------------------------------

def compare_and_save(
    baseline_results,
    xgb_results,
    X_validation,
    y_validation,
    y_test,
):
    """Compare models and save the winner."""

    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)

    print(
        f"{'Model':<35}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>10}"
        f"{'PR-AUC':>10}"
    )

    print("-" * 85)

    for result in [
        baseline_results,
        xgb_results,
    ]:
        print(
            f"{result['model_name']:<35}"
            f"{result['precision']:>10.4f}"
            f"{result['recall']:>10.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['roc_auc']:>10.4f}"
            f"{result['pr_auc']:>10.4f}"
        )

    winner = max(
        [
            baseline_results,
            xgb_results,
        ],
        key=lambda result: result["f1"],
    )

    print(
        f"\nWinner: "
        f"{winner['model_name']} "
        f"(F1: {winner['f1']:.4f})"
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR /
        "bhairava_model.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as file:
        pickle.dump(
            winner["model"],
            file,
        )

    print(
        f"Model saved: {model_path}"
    )

    # -----------------------------------------------------
    # Save validation probabilities
    # -----------------------------------------------------

    validation_proba = (
        winner["model"]
        .predict_proba(X_validation)[:, 1]
    )

    validation_proba_path = (
        MODEL_DIR /
        "bhairava_validation_probas.npy"
    )

    np.save(
        validation_proba_path,
        validation_proba,
    )

    print(
        f"Validation probabilities saved: "
        f"{validation_proba_path}"
    )

    # -----------------------------------------------------
    # Save validation labels
    # -----------------------------------------------------

    validation_labels_path = (
        MODEL_DIR /
        "bhairava_validation_labels.npy"
    )

    np.save(
        validation_labels_path,
        y_validation.to_numpy(),
    )

    print(
        f"Validation labels saved: "
        f"{validation_labels_path}"
    )

    # -----------------------------------------------------
    # Save test probabilities
    # -----------------------------------------------------

    test_proba_path = (
        MODEL_DIR /
        "bhairava_test_probas.npy"
    )

    np.save(
        test_proba_path,
        winner["y_proba"],
    )

    print(
        f"Test probabilities saved: "
        f"{test_proba_path}"
    )

    # -----------------------------------------------------
    # Save test labels
    # -----------------------------------------------------

    test_labels_path = (
        MODEL_DIR /
        "bhairava_test_labels.npy"
    )

    np.save(
        test_labels_path,
        y_test.to_numpy(),
    )

    print(
        f"Test labels saved: "
        f"{test_labels_path}"
    )

    return winner


# ---------------------------------------------------------
# Master training pipeline
# ---------------------------------------------------------

def train():
    """Run the complete Bhairava training pipeline."""

    print("=" * 60)
    print("Bhairava Training Pipeline")
    print("=" * 60)

    # 1. Load + preprocess
    df = get_clean_data()

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Chronological split
    (
        train_df,
        validation_df,
        test_df,
    ) = time_based_split(df)

    # 4. Separate features + target
    X_train, y_train = (
        get_features_and_target(train_df)
    )

    X_validation, y_validation = (
        get_features_and_target(validation_df)
    )

    X_test, y_test = (
        get_features_and_target(test_df)
    )

    print(
        f"\nFeature matrix:"
        f"\n  Train:      {X_train.shape}"
        f"\n  Validation: {X_validation.shape}"
        f"\n  Test:       {X_test.shape}"
    )

    # 5. Train baseline
    baseline_results = train_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # 6. Train XGBoost
    xgb_results = train_xgboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    # 7. Compare + save winner
    winner = compare_and_save(
        baseline_results,
        xgb_results,
        X_validation,
        y_validation,
        y_test,
    )

    print("\n" + "=" * 60)
    print("Training complete.")
    print("Next: threshold optimization.")
    print("=" * 60)

    return winner


if __name__ == "__main__":
    train()