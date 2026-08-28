"""
Bhairava — Fraud Detection System
ml/evaluation/threshold.py

Analyzes fraud probability thresholds using the
validation set and locks the threshold that maximizes F1.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data/models"


# ---------------------------------------------------------
# Load predictions
# ---------------------------------------------------------

def load_predictions():
    """Load validation probabilities and labels."""

    validation_proba = np.load(
        MODEL_DIR / "bhairava_validation_probas.npy"
    )

    validation_labels = np.load(
        MODEL_DIR / "bhairava_validation_labels.npy"
    )

    return validation_proba, validation_labels


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

def analyze_thresholds(
    y_true,
    probabilities,
):
    """
    Evaluate precision, recall, and F1 across
    probability thresholds.
    """

    results = []

    for threshold in np.arange(
        0.05,
        0.96,
        0.01,
    ):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        flagged = int(
            predictions.sum()
        )

        results.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flagged": flagged,
            }
        )

    return results


# ---------------------------------------------------------
# Find best threshold
# ---------------------------------------------------------

def find_best_threshold(results):
    """
    Select the threshold with the highest
    validation F1 score.
    """

    best = max(
        results,
        key=lambda result: result["f1"],
    )

    return best


# ---------------------------------------------------------
# Display operating points
# ---------------------------------------------------------

def display_operating_points(results):
    """Display useful threshold operating points."""

    print("\n" + "=" * 80)
    print("Threshold Operating Points")
    print("=" * 80)

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Flagged':>12}"
    )

    print("-" * 80)

    for result in results:
        threshold = result["threshold"]

        if (
            abs(threshold - 0.20) < 0.001
            or abs(threshold - 0.30) < 0.001
            or abs(threshold - 0.40) < 0.001
            or abs(threshold - 0.50) < 0.001
            or abs(threshold - 0.59) < 0.001
            or abs(threshold - 0.60) < 0.001
            or abs(threshold - 0.70) < 0.001
            or abs(threshold - 0.80) < 0.001
        ):
            print(
                f"{threshold:>10.2f}"
                f"{result['precision']:>12.4f}"
                f"{result['recall']:>12.4f}"
                f"{result['f1']:>12.4f}"
                f"{result['flagged']:>12,}"
            )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("Bhairava Threshold Optimization")
    print("=" * 60)

    # Load validation data only
    validation_proba, validation_labels = (
        load_predictions()
    )

    print(
        f"\nValidation samples: "
        f"{len(validation_labels):,}"
    )

    # Analyze threshold range
    results = analyze_thresholds(
        validation_labels,
        validation_proba,
    )

    # Display useful operating points
    display_operating_points(results)

    # Find maximum-F1 threshold
    best = find_best_threshold(results)

    print("\n" + "=" * 60)
    print("Optimal Threshold")
    print("=" * 60)

    print(
        f"Best threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Validation Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Validation Recall:    "
        f"{best['recall']:.4f}"
    )

    print(
        f"Validation F1:        "
        f"{best['f1']:.4f}"
    )

    print(
        f"Transactions flagged: "
        f"{best['flagged']:,}"
    )

    # -----------------------------------------------------
    # Save locked threshold
    # -----------------------------------------------------

    threshold_path = (
        MODEL_DIR /
        "bhairava_threshold.json"
    )

    with open(
        threshold_path,
        "w",
    ) as file:
        json.dump(
            {
                "threshold": best["threshold"],
                "validation_precision": best[
                    "precision"
                ],
                "validation_recall": best[
                    "recall"
                ],
                "validation_f1": best[
                    "f1"
                ],
                "validation_flagged": best[
                    "flagged"
                ],
            },
            file,
            indent=4,
        )

    print(
        f"\nThreshold saved: "
        f"{threshold_path}"
    )

    print("\nThreshold locked.")

    print("\nNext step:")
    print(
        "Evaluate the locked threshold "
        "on the untouched test set."
    )


if __name__ == "__main__":
    main()