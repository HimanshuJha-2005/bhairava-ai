"""
Bhairava — Fraud Detection System
ml/features/feature_engineering.py

Engineers fraud-specific features from clean transaction data.

Principle: A smart feature beats a complex model.
The V1-V338 columns in IEEE-CIS are already Vesta's engineered
features. These are OUR additions on top — domain-driven signals
that come from understanding how fraud actually behaves.
"""

import pandas as pd
import numpy as np


def add_time_features(df):
    """
    TransactionDT is seconds elapsed from a reference point (not a real timestamp).
    We extract hour and day of week from it.

    Why this matters:
    Fraud doesn't happen uniformly. Stolen cards get used at odd hours.
    Late night / early morning transactions have higher fraud rates.
    Fraudsters also exploit weekends when bank fraud teams are smaller.
    """
    print("  Adding time features...")

    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

    # Late night flag (midnight to 6am) — higher fraud risk window
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(np.int8)

    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)

    return df


def add_amount_features(df):
    """
    Amount anomaly detection.

    Raw amount alone isn't enough — we need context:
    - How unusual is this amount globally? (z-score)
    - How unusual is it for THIS specific card? (ratio to card average)
    - Is it suspiciously round? (fraudsters often test with exact amounts)
    - Log transform for skewed distribution (XGBoost handles this better)
    """
    print("  Adding amount features...")

    # How many standard deviations from average
    mean_amt = df['TransactionAmt'].mean()
    std_amt = df['TransactionAmt'].std()
    df['amount_zscore'] = (
        (df['TransactionAmt'] - mean_amt) / std_amt
    )

    # Log transform — transaction amounts are heavily right-skewed
    df['amount_log'] = np.log1p(df['TransactionAmt'])

    # Suspiciously round amounts (e.g., exactly 100.00, 500.00)
    df['is_round_amount'] = (
        df['TransactionAmt'] % 1 == 0
    ).astype(np.int8)

    # Amount relative to this card's historical average
    # This catches transactions that are unusually large for a card.

    card_count = df.groupby('card1').cumcount()

    card_sum = (
        df.groupby('card1')['TransactionAmt'].cumsum()
        - df['TransactionAmt']
    )

    card_mean = card_sum / card_count.replace(0, np.nan)

    # For the first transaction of a card, no historical average exists.
    # Use the current amount as the neutral baseline → ratio = 1.
    card_mean = card_mean.fillna(df['TransactionAmt'])

    df['amount_vs_card_mean'] = (
        df['TransactionAmt'] / (card_mean + 1)
    )

    return df


def add_velocity_features(df):
    """
    Historical transaction-frequency signals.

    Every feature only uses transactions that occurred BEFORE
    the current transaction, preventing temporal leakage.
    """
    print("  Adding velocity features...")

    # Data must be chronological
    df = df.sort_values('TransactionDT').copy()

    # Number of previous transactions for this card
    df['card1_txn_count'] = (
        df.groupby('card1').cumcount()
    )

    # Number of previous transactions for this card + address
    df['card_addr_txn_count'] = (
        df.groupby(['card1', 'addr1']).cumcount()
    )

    # Number of previous transactions for this email domain
    df['email_txn_count'] = (
        df.groupby('P_emaildomain').cumcount()
    )

    return df


def add_card_features(df):
    """
    Card-based risk signals.

    Multiple cards from the same billing address = suspicious
    (card testing, where fraudsters try multiple stolen cards).

    High amount + card rarely seen before = classic stolen card pattern.
    """
    print("  Adding card features...")

    # How many unique cards have been used from this billing address
    df['unique_cards_per_addr'] = df.groupby('addr1')['card1'].transform('nunique')

    # Flag: high amount transaction on a rarely-used card
    high_amount_threshold = df['TransactionAmt'].quantile(0.90)
    df['high_amount_new_card'] = (
        (df['TransactionAmt'] > high_amount_threshold) &
        (df['card1_txn_count'] < 5)
    ).astype(np.int8)

    return df


def add_email_features(df):
    """
    Email domain risk signals.

    Rare email domains are more suspicious — fraudsters
    often use newly created or unusual email domains.

    Low frequency domain in this dataset = unusual = flag it.
    """
    print("  Adding email features...")

    # How many times has this email domain appeared previously?
    df['email_domain_freq'] = (
        df.groupby('P_emaildomain').cumcount()
    )

    # Rare domain flag
    # Appears fewer than 100 times before this transaction.
    df['is_rare_email_domain'] = (
        df['email_domain_freq'] < 100
    ).astype(np.int8)

    return df


def add_device_features(df):
    """
    Device-based signals from the identity file.

    Missing device info is itself a signal (no DeviceType = no identity data).
    Mobile transactions have slightly different fraud patterns than desktop.
    """
    print("  Adding device features...")

    

    return df


def engineer_features(df):
    """
    Master function — runs all feature engineering steps.

    Call this after preprocessing.get_clean_data().
    Returns dataframe with all Bhairava fraud signals added.
    """
    print("\n" + "=" * 50)
    print("Bhairava Feature Engineering")
    print("=" * 50)

    original_cols = df.shape[1]

    # Fraud features must be generated chronologically
    # to prevent future transactions from leaking into features.
    df = df.sort_values('TransactionDT').copy()

    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = add_card_features(df)
    df = add_email_features(df)
    df = add_device_features(df)

    new_cols = df.shape[1] - original_cols

    print("\n" + "=" * 50)
    print("✅ Features ready")
    print(f"   Original columns:  {original_cols}")
    print(f"   New features added: {new_cols}")
    print(f"   Total columns:     {df.shape[1]}")
    print("=" * 50)

    return df


def get_feature_list():
    """
    Returns the list of features Bhairava engineered.
    Useful for model training and explainability.
    """
    return [
        # Time features
        'hour', 'day_of_week', 'is_night', 'is_weekend',
        # Amount features
        'amount_zscore', 'amount_log', 'is_round_amount', 'amount_vs_card_mean',
        # Velocity features
        'card1_txn_count', 'card_addr_txn_count', 'email_txn_count',
        # Card features
        'unique_cards_per_addr', 'high_amount_new_card',
        # Email features
        'email_domain_freq', 'is_rare_email_domain',
        # Device features
        'has_identity_data',
    ]


if __name__ == "__main__":
    import sys
    sys.path.append(str(__file__.split('ml')[0]))

    from ml.data.preprocessing import get_clean_data

    df = get_clean_data()
    df = engineer_features(df)

    print("\nNew features sample:")
    print(df[get_feature_list()].describe().round(3))

    print("\nFraud vs Legit — key feature averages:")
    features_to_check = ['is_night', 'amount_zscore', 'high_amount_new_card', 'is_rare_email_domain']
    print(df.groupby('isFraud')[features_to_check].mean().round(3))