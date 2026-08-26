"""
Bhairava — Fraud Detection System
ml/evaluation/threshold.py

Finds the optimal fraud probability threshold using
the validation set, then evaluates that locked threshold
on the untouched test set.
"""

import json
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"


def load_predictions():
    """Load saved validation and test predictions."""

    validation_proba = np.load(
        MODEL_DIR / "bhairava_validation_probas.npy"
    )

    validation_labels = np.load(
        MODEL_DIR / "bhairava_validation_labels.npy"
    )

    test_proba = np.load(
        MODEL_DIR / "bhairava_test_probas.npy"
    )

    return validation_proba, validation_labels, test_proba


def find_best_threshold(y_true, probabilities):
    """
    Search probability thresholds and select the one
    producing the highest validation F1 score.
    """

    best_threshold = 0.5
    best_f1 = 0.0

    results = []

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        results.append({
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1, results


def evaluate_threshold(y_true, probabilities, threshold):
    """Evaluate a locked threshold."""

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("Final Test Results")
    print("=" * 60)

    print(f"Threshold:  {threshold:.2f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    print("=" * 60)
    print("Bhairava Threshold Optimization")
    print("=" * 60)

    validation_proba, validation_labels, test_proba = load_predictions()

    print(f"\nValidation samples: {len(validation_labels):,}")
    print(f"Test samples:       {len(test_proba):,}")

    # Optimize ONLY on validation data
    best_threshold, best_f1, threshold_results = find_best_threshold(
        validation_labels,
        validation_proba
    )

    print("\n" + "=" * 60)
    print("Optimal Threshold")
    print("=" * 60)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation F1:  {best_f1:.4f}")

    # Save threshold for the risk engine
    threshold_path = MODEL_DIR / "bhairava_threshold.json"

    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "validation_f1": best_f1
            },
            f,
            indent=4
        )

    print(f"Threshold saved: {threshold_path}")

    # IMPORTANT:
    # Test labels are intentionally not used here because
    # the test set must remain untouched for final evaluation.
    print("\nThreshold locked.")

    print("\nNext step:")
    print("Evaluate the locked threshold on the test set.")


if __name__ == "__main__":
    main()