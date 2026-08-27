"""
Bhairava — Fraud Detection System
ml/evaluation/threshold.py

Finds the optimal fraud probability threshold using
the validation set and locks it for final evaluation.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"


def load_predictions():
    """Load validation probabilities and labels."""

    validation_proba = np.load(
        MODEL_DIR / "bhairava_validation_probas.npy"
    )

    validation_labels = np.load(
        MODEL_DIR / "bhairava_validation_labels.npy"
    )

    return validation_proba, validation_labels


def find_best_threshold(y_true, probabilities):
    """
    Search probability thresholds and select the threshold
    that produces the highest validation F1 score.
    """

    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def main():

    print("=" * 60)
    print("Bhairava Threshold Optimization")
    print("=" * 60)

    # Load validation data only
    validation_proba, validation_labels = load_predictions()

    print(
        f"\nValidation samples: "
        f"{len(validation_labels):,}"
    )

    # Optimize threshold ONLY on validation data
    best_threshold, best_f1 = find_best_threshold(
        validation_labels,
        validation_proba
    )

    print("\n" + "=" * 60)
    print("Optimal Threshold")
    print("=" * 60)

    print(
        f"Best threshold: {best_threshold:.2f}"
    )

    print(
        f"Validation F1:  {best_f1:.4f}"
    )

    # Save locked threshold
    threshold_path = (
        MODEL_DIR / "bhairava_threshold.json"
    )

    with open(threshold_path, "w") as file:
        json.dump(
            {
                "threshold": best_threshold,
                "validation_f1": best_f1
            },
            file,
            indent=4
        )

    print(
        f"Threshold saved: {threshold_path}"
    )

    print("\nThreshold locked.")

    print("\nNext step:")
    print(
        "Evaluate the locked threshold "
        "on the untouched test set."
    )


if __name__ == "__main__":
    main()