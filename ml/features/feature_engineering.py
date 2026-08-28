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

    # Combined temporal signal
    df["is_night_weekend"] = (
        (df["is_night"] == 1)
        & (df["is_weekend"] == 1)
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
    # Round amount
    # -----------------------------------------------------

    df["is_round_amount"] = (
        np.isclose(
            amount % 1,
            0,
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # Historical card amount behavior
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

    # -----------------------------------------------------
    # Historical address amount behavior
    # -----------------------------------------------------

    addr_count = (
        df.groupby("addr1")
        .cumcount()
    )

    addr_sum = (
        df.groupby("addr1")["TransactionAmt"]
        .cumsum()
        - amount
    )

    addr_mean = (
        addr_sum
        / addr_count.replace(
            0,
            np.nan,
        )
    )

    addr_mean = addr_mean.fillna(
        amount
    )

    df["amount_vs_addr_mean"] = (
        amount
        / (addr_mean + 1e-6)
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
    # Card velocity
    # -----------------------------------------------------

    df["card1_txn_count"] = (
        df.groupby("card1")
        .cumcount()
    ).astype(np.int32)

    # -----------------------------------------------------
    # Card + address velocity
    # -----------------------------------------------------

    df["card_addr_txn_count"] = (
        df.groupby(
            ["card1", "addr1"]
        )
        .cumcount()
    ).astype(np.int32)

    # -----------------------------------------------------
    # Email velocity
    # -----------------------------------------------------

    df["email_txn_count"] = (
        df.groupby("P_emaildomain")
        .cumcount()
    ).astype(np.int32)

    # -----------------------------------------------------
    # Device velocity
    # -----------------------------------------------------

    if "DeviceType" in df.columns:

        df["device_txn_count"] = (
            df.groupby("DeviceType")
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
    # Very rapid card reuse
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

    # -----------------------------------------------------
    # Historical unique cards per address
    # -----------------------------------------------------

    # Mark the first occurrence of each address/card pair.
    #
    # Example:
    #
    # addr A + card 1  -> first occurrence = 1
    # addr A + card 1  -> first occurrence = 0
    # addr A + card 2  -> first occurrence = 1
    #
    # The cumulative sum BEFORE the current row therefore
    # represents how many distinct cards were previously
    # observed at this address.

    new_card_addr_pair = (
        ~df.duplicated(
            subset=["addr1", "card1"],
            keep="first",
        )
    ).astype(np.int8)

    historical_unique_cards = (
        new_card_addr_pair
        .groupby(df["addr1"])
        .cumsum()
        - new_card_addr_pair
    )

    # Count distinct cards previously seen at each billing address.
    # The dataframe is chronological, so the first occurrence of
    # each (addr1, card1) pair represents a newly observed card.
    is_new_card = ~df.duplicated(
        subset=['addr1', 'card1'],
        keep='first',
    )

    df['_new_card_at_addr'] = is_new_card.astype(np.int8)

    df['unique_cards_per_addr'] = (
        df.groupby('addr1')['_new_card_at_addr'].cumsum()
            - df['_new_card_at_addr']
    )

    df.drop(columns=['_new_card_at_addr'], inplace=True)

    # -----------------------------------------------------
    # Historical high-amount behavior
    # -----------------------------------------------------

    # Instead of using the full-dataset 90th percentile,
    # which would leak future information, use a historical
    # mean + 2 standard deviations threshold.

    historical_amount_threshold = (
        df["TransactionAmt"]
        .expanding(min_periods=50)
        .mean()
        .shift(1)
        +
        2
        *
        df["TransactionAmt"]
        .expanding(min_periods=50)
        .std()
        .shift(1)
    )

    historical_amount_threshold = (
        historical_amount_threshold
        .fillna(
            df["TransactionAmt"]
        )
    )

    # -----------------------------------------------------
    # High amount + rarely used card
    # -----------------------------------------------------

    df["high_amount_new_card"] = (
        (
            df["TransactionAmt"]
            > historical_amount_threshold
        )
        & (
            df["card1_txn_count"] < 5
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # New card/address combination
    # -----------------------------------------------------

    df["is_new_card_addr"] = (
        df["card_addr_txn_count"] == 0
    ).astype(np.int8)

    # -----------------------------------------------------
    # Card reuse intensity
    # -----------------------------------------------------

    df["card_reuse_signal"] = (
        np.log1p(
            df["card1_txn_count"]
        )
    ).astype(np.float32)

    # -----------------------------------------------------
    # Address card diversity
    # -----------------------------------------------------

    df["multiple_cards_same_addr"] = (
        df["unique_cards_per_addr"] >= 2
    ).astype(np.int8)

    return df


# ---------------------------------------------------------
# Email Features
# ---------------------------------------------------------

def add_email_features(df):
    """
    Historical email-domain risk signals.
    """

    print("  Adding email features...")

    df["email_domain_freq"] = (
        df.groupby("P_emaildomain")
        .cumcount()
    ).astype(np.int32)

    # Rare domain
    df["is_rare_email_domain"] = (
        df["email_domain_freq"] < 100
    ).astype(np.int8)

    # First observed transaction for domain
    df["is_new_email_domain"] = (
        df["email_domain_freq"] == 0
    ).astype(np.int8)

    return df


# ---------------------------------------------------------
# Device Features
# ---------------------------------------------------------

def add_device_features(df):
    """
    Device and identity availability signals.

    has_identity_data is created during preprocessing.
    """

    print("  Adding device features...")

    if "has_identity_data" in df.columns:

        df["missing_identity_data"] = (
            df["has_identity_data"] == 0
        ).astype(np.int8)

    return df


# ---------------------------------------------------------
# Interaction Features
# ---------------------------------------------------------

def add_interaction_features(df):
    """
    Combine individual fraud signals into higher-level
    behavioral indicators.

    These features do not use the target variable.
    """

    print("  Adding interaction features...")

    # -----------------------------------------------------
    # Large amount relative to card history + new card
    # -----------------------------------------------------

    df["amount_new_card_interaction"] = (
        (
            df["amount_vs_card_mean"] > 2.0
        )
        & (
            df["card1_txn_count"] < 5
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # New card + new email
    # -----------------------------------------------------

    df["new_card_new_email"] = (
        (
            df["card1_txn_count"] == 0
        )
        & (
            df["email_domain_freq"] == 0
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # Night + unusually large amount
    # -----------------------------------------------------

    df["night_high_amount"] = (
        (
            df["is_night"] == 1
        )
        & (
            df["amount_zscore"] > 2
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # Missing identity + new card
    # -----------------------------------------------------

    if "missing_identity_data" in df.columns:

        df["missing_identity_new_card"] = (
            (
                df["missing_identity_data"] == 1
            )
            & (
                df["card1_txn_count"] < 5
            )
        ).astype(np.int8)

    # -----------------------------------------------------
    # Rapid activity + high amount
    # -----------------------------------------------------

    df["rapid_high_amount"] = (
        (
            df["rapid_card_activity"] == 1
        )
        & (
            df["amount_vs_card_mean"] > 2
        )
    ).astype(np.int8)

    # -----------------------------------------------------
    # Multiple cards + missing identity
    # -----------------------------------------------------

    if "missing_identity_data" in df.columns:

        df["multiple_cards_missing_identity"] = (
            (
                df["multiple_cards_same_addr"] == 1
            )
            & (
                df["missing_identity_data"] == 1
            )
        ).astype(np.int8)

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
    df = add_device_features(df)
    df = add_interaction_features(df)

    new_cols = (
        df.shape[1]
        - original_cols
    )

    print("\n" + "=" * 50)
    print("✅ Features ready")
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
        "is_night_weekend",

        # Amount
        "amount_zscore",
        "amount_log",
        "is_round_amount",
        "amount_vs_card_mean",
        "amount_vs_card_addr_mean",
        "amount_vs_addr_mean",

        # Velocity
        "card1_txn_count",
        "card_addr_txn_count",
        "email_txn_count",
        "device_txn_count",
        "card_time_since_prev",
        "card_addr_time_since_prev",
        "rapid_card_activity",

        # Card
        "unique_cards_per_addr",
        "high_amount_new_card",
        "is_new_card_addr",
        "card_reuse_signal",
        "multiple_cards_same_addr",

        # Email
        "email_domain_freq",
        "is_rare_email_domain",
        "is_new_email_domain",

        # Device
        "has_identity_data",
        "missing_identity_data",

        # Interactions
        "amount_new_card_interaction",
        "new_card_new_email",
        "night_high_amount",
        "missing_identity_new_card",
        "rapid_high_amount",
        "multiple_cards_missing_identity",
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
        "amount_zscore",
        "amount_vs_card_mean",
        "card1_txn_count",
        "high_amount_new_card",
        "is_rare_email_domain",
        "missing_identity_data",
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