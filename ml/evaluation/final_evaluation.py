"""
Bhairava — Fraud Detection System
ml/evaluation/final_evaluation.py

Evaluates the locked fraud threshold on the untouched test set.
"""

import json
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"


def load_data():
    """Load locked threshold and test predictions."""

    with open(
        MODEL_DIR / "bhairava_threshold.json",
        "r"
    ) as file:
        threshold_data = json.load(file)

    threshold = threshold_data["threshold"]

    test_proba = np.load(
        MODEL_DIR / "bhairava_test_probas.npy"
    )

    test_labels = np.load(
        MODEL_DIR / "bhairava_test_labels.npy"
    )

    return threshold, test_proba, test_labels


def evaluate(threshold, probabilities, labels):
    """Evaluate the locked threshold on the untouched test set."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    pr_auc = average_precision_score(
        labels,
        probabilities
    )

    print("\n" + "=" * 60)
    print("Bhairava Final Test Evaluation")
    print("=" * 60)

    print(f"Locked threshold: {threshold:.2f}")
    print(f"Test samples:     {len(labels):,}")

    print("\nFinal Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    results = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    results_path = (
        MODEL_DIR /
        "bhairava_final_results.json"
    )

    with open(results_path, "w") as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"Results saved: {results_path}"
    )


def main():

    threshold, test_proba, test_labels = load_data()

    evaluate(
        threshold,
        test_proba,
        test_labels
    )


if __name__ == "__main__":
    main()