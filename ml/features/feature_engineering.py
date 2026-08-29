"""
Bhairava — Fraud Detection System
ml/features/feature_engineering.py

Engineers fraud-specific features from clean transaction data.

Principle:
A smart feature beats a complex model.

All historical features are calculated chronologically.
A transaction must never use information from transactions
that occur after it.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Time Features
# ---------------------------------------------------------

def add_time_features(df):
    """
    Extract temporal fraud signals from TransactionDT.

    TransactionDT represents elapsed seconds from a reference
    point rather than a real-world timestamp.
    """

    print("  Adding time features...")

    df["hour"] = (
        (df["TransactionDT"] // 3600) % 24
    ).astype(np.int8)

    df["day_of_week"] = (
        (df["TransactionDT"] // (3600 * 24)) % 7
    ).astype(np.int8)

    # Midnight to 6 AM
    df["is_night"] = (
        df["hour"] < 6
    ).astype(np.int8)

    # Saturday and Sunday
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(np.int8)

    return df


# ---------------------------------------------------------
# Amount Features
# ---------------------------------------------------------

def add_amount_features(df):
    """
    Create transaction amount anomaly features.

    Historical statistics use only transactions that occurred
    before the current transaction.
    """

    print("  Adding amount features...")

    amount = df["TransactionAmt"].astype(np.float64)

    # -----------------------------------------------------
    # Historical global amount statistics
    # -----------------------------------------------------

    transaction_count = np.arange(
        len(df),
        dtype=np.float64,
    )

    cumulative_sum = (
        amount.cumsum() - amount
    )

    cumulative_squared_sum = (
        (amount ** 2).cumsum()
        - (amount ** 2)
    )

    historical_mean = (
        cumulative_sum
        / np.maximum(transaction_count, 1)
    )

    historical_variance = (
        cumulative_squared_sum
        / np.maximum(transaction_count, 1)
        - historical_mean ** 2
    )

    historical_variance = (
        historical_variance
        .clip(lower=0)
    )

    historical_std = np.sqrt(
        historical_variance
    )

    # First transaction has no history.
    historical_mean = (
        historical_mean
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(amount)
    )

    historical_std = (
        historical_std
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(1.0)
        .clip(lower=1e-6)
    )

    df["amount_zscore"] = (
        (
            amount - historical_mean
        )
        / historical_std
    ).clip(-20, 20).astype(np.float32)

    # -----------------------------------------------------
    # Log amount
    # -----------------------------------------------------

    df["amount_log"] = (
        np.log1p(amount)
    ).astype(np.float32)

    # -----------------------------------------------------
    # Amount cents / fractional component
    # -----------------------------------------------------

    df["amount_decimal"] = (
        amount - np.floor(amount)
    ).astype(np.float32)

    # -----------------------------------------------------
    # Round amount
    # -----------------------------------------------------

    df["is_round_amount"] = (
        np.isclose(
            amount % 1,
            0,
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # Historical card amount behavior (card1)
    # -----------------------------------------------------

    card_count = (
        df.groupby("card1")
        .cumcount()
    )

    card_sum = (
        df.groupby("card1")["TransactionAmt"]
        .cumsum()
        - amount
    )

    card_mean = (
        card_sum
        / card_count.replace(
            0,
            np.nan,
        )
    )

    card_mean = card_mean.fillna(
        amount
    )

    df["amount_vs_card_mean"] = (
        amount
        / (card_mean + 1e-6)
    ).clip(
        0,
        100,
    ).astype(np.float32)

    # -----------------------------------------------------
    # Historical card + address amount behavior
    # -----------------------------------------------------

    if "addr1" in df.columns:
        card_addr_count = (
            df.groupby(
                ["card1", "addr1"]
            )
            .cumcount()
        )

        card_addr_sum = (
            df.groupby(
                ["card1", "addr1"]
            )["TransactionAmt"]
            .cumsum()
            - amount
        )

        card_addr_mean = (
            card_addr_sum
            / card_addr_count.replace(
                0,
                np.nan,
            )
        )

        card_addr_mean = card_addr_mean.fillna(
            amount
        )

        df["amount_vs_card_addr_mean"] = (
            amount
            / (card_addr_mean + 1e-6)
        ).clip(
            0,
            100,
        ).astype(np.float32)

    return df


# ---------------------------------------------------------
# Velocity Features
# ---------------------------------------------------------

def add_velocity_features(df):
    """
    Historical transaction-frequency and timing signals.

    Every feature only uses transactions that occurred before
    the current transaction.
    """

    print("  Adding velocity features...")

    # -----------------------------------------------------
    # Card velocity (card1)
    # -----------------------------------------------------

    df["card1_txn_count"] = (
        df.groupby("card1")
        .cumcount()
    ).astype(np.int32)

    # -----------------------------------------------------
    # Card + address velocity
    # -----------------------------------------------------

    if "addr1" in df.columns:
        df["card_addr_txn_count"] = (
            df.groupby(
                ["card1", "addr1"]
            )
            .cumcount()
        ).astype(np.int32)

    # -----------------------------------------------------
    # Composite card identity velocity
    # -----------------------------------------------------

    card_composite_cols = [
        col for col in ["card1", "card2", "card3", "card5", "addr1"]
        if col in df.columns
    ]

    if len(card_composite_cols) > 1:
        df["card_full_txn_count"] = (
            df.groupby(card_composite_cols)
            .cumcount()
        ).astype(np.int32)

    # -----------------------------------------------------
    # Email velocity
    # -----------------------------------------------------

    if "P_emaildomain" in df.columns:
        df["email_txn_count"] = (
            df.groupby("P_emaildomain")
            .cumcount()
        ).astype(np.int32)

    # -----------------------------------------------------
    # Time since previous card transaction
    # -----------------------------------------------------

    df["card_time_since_prev"] = (
        df.groupby("card1")["TransactionDT"]
        .diff()
        .fillna(-1)
        .clip(lower=-1)
        .astype(np.float32)
    )

    # -----------------------------------------------------
    # Time since previous card + address transaction
    # -----------------------------------------------------

    if "addr1" in df.columns:
        df["card_addr_time_since_prev"] = (
            df.groupby(
                ["card1", "addr1"]
            )["TransactionDT"]
            .diff()
            .fillna(-1)
            .clip(lower=-1)
            .astype(np.float32)
        )

    # -----------------------------------------------------
    # Very rapid card reuse (< 1 hour)
    # -----------------------------------------------------

    df["rapid_card_activity"] = (
        (
            df["card_time_since_prev"] >= 0
        )
        & (
            df["card_time_since_prev"] < 3600
        )
    ).astype(np.int8)

    return df


# ---------------------------------------------------------
# Card Features
# ---------------------------------------------------------

def add_card_features(df):
    """
    Card/address relationship signals.

    All relationship features are based only on transactions
    already observed.
    """

    print("  Adding card features...")

    if "addr1" in df.columns:
        # Count distinct cards previously seen at each billing address.
        # Single-pass chronological calculation without leakage.
        is_new_card = (
            ~df.duplicated(
                subset=["addr1", "card1"],
                keep="first",
            )
        ).astype(np.int32)

        df["unique_cards_per_addr"] = (
            is_new_card.groupby(df["addr1"]).cumsum()
            - is_new_card
        ).astype(np.int32)

    # -----------------------------------------------------
    # New card/address combination
    # -----------------------------------------------------

    if "card_addr_txn_count" in df.columns:
        df["is_new_card_addr"] = (
            df["card_addr_txn_count"] == 0
        ).astype(np.int8)

    # -----------------------------------------------------
    # Card reuse intensity (log scale)
    # -----------------------------------------------------

    df["card_reuse_signal"] = (
        np.log1p(
            df["card1_txn_count"]
        )
    ).astype(np.float32)

    return df


# ---------------------------------------------------------
# Email Features
# ---------------------------------------------------------

def add_email_features(df):
    """
    Historical email-domain risk signals.
    """

    print("  Adding email features...")

    if "P_emaildomain" in df.columns:
        df["email_domain_freq"] = (
            df.groupby("P_emaildomain")
            .cumcount()
        ).astype(np.int32)

        # Rare domain
        df["is_rare_email_domain"] = (
            df["email_domain_freq"] < 100
        ).astype(np.int8)

    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (
            df["P_emaildomain"] == df["R_emaildomain"]
        ).astype(np.int8)

    return df


# ---------------------------------------------------------
# Temporal Anchor Features
# ---------------------------------------------------------

def add_temporal_anchor_features(df):
    """
    Card age anchor features from Vesta D-columns.

    D1 represents timedelta from card inception/first transaction.
    (TransactionDT / 86400) - D1 calculates the historical card
    anchor day at transaction time without future leakage.
    """

    print("  Adding temporal anchor features...")

    if "D1" in df.columns:
        df["D1_anchor_day"] = (
            (df["TransactionDT"] / 86400.0)
            - df["D1"]
        ).astype(np.float32)

    if "D2" in df.columns:
        df["D2_anchor_day"] = (
            (df["TransactionDT"] / 86400.0)
            - df["D2"]
        ).astype(np.float32)

    return df


# ---------------------------------------------------------
# Master Feature Engineering
# ---------------------------------------------------------

def engineer_features(df):
    """
    Run the complete feature-engineering pipeline.

    The dataframe is sorted chronologically BEFORE any
    historical feature is calculated.
    """

    print("\n" + "=" * 50)
    print("Bhairava Feature Engineering")
    print("=" * 50)

    original_cols = df.shape[1]

    # -----------------------------------------------------
    # CRITICAL:
    # Historical features require chronological ordering.
    # -----------------------------------------------------

    df = (
        df.sort_values("TransactionDT")
        .reset_index(drop=True)
        .copy()
    )

    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = add_card_features(df)
    df = add_email_features(df)
    df = add_temporal_anchor_features(df)

    new_cols = (
        df.shape[1]
        - original_cols
    )

    print("\n" + "=" * 50)
    print("[+] Features ready")
    print(
        f"   Original columns:   "
        f"{original_cols}"
    )
    print(
        f"   New features added: "
        f"{new_cols}"
    )
    print(
        f"   Total columns:      "
        f"{df.shape[1]}"
    )
    print("=" * 50)

    return df


# ---------------------------------------------------------
# Feature List
# ---------------------------------------------------------

def get_feature_list():
    """
    Return Bhairava's engineered feature names.
    """

    return [
        # Time
        "hour",
        "day_of_week",
        "is_night",
        "is_weekend",

        # Amount
        "amount_zscore",
        "amount_log",
        "amount_decimal",
        "is_round_amount",
        "amount_vs_card_mean",
        "amount_vs_card_addr_mean",

        # Velocity
        "card1_txn_count",
        "card_addr_txn_count",
        "card_full_txn_count",
        "email_txn_count",
        "card_time_since_prev",
        "card_addr_time_since_prev",
        "rapid_card_activity",

        # Card
        "unique_cards_per_addr",
        "is_new_card_addr",
        "card_reuse_signal",

        # Email
        "email_domain_freq",
        "is_rare_email_domain",
        "email_domain_match",

        # Temporal anchor
        "D1_anchor_day",
        "D2_anchor_day",
    ]


# ---------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys
    from pathlib import Path

    sys.path.append(
        str(
            Path(__file__).resolve().parents[2]
        )
    )

    from ml.data.preprocessing import (
        get_clean_data
    )

    df = get_clean_data()

    df = engineer_features(df)

    print("\nEngineered feature summary:")

    available_features = [
        feature
        for feature in get_feature_list()
        if feature in df.columns
    ]

    print(
        df[available_features]
        .describe()
        .round(3)
    )

    print("\nFraud vs Legitimate:")

    features_to_check = [
        "is_night",
        "is_weekend",
        "amount_decimal",
        "amount_zscore",
        "amount_vs_card_mean",
        "card1_txn_count",
        "is_rare_email_domain",
        "rapid_card_activity",
    ]

    available_check_features = [
        feature
        for feature in features_to_check
        if feature in df.columns
    ]

    print(
        df.groupby("isFraud")[
            available_check_features
        ]
        .mean()
        .round(3)
    )