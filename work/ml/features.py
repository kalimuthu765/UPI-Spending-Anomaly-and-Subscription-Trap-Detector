"""
Phase 3 — Feature Engineering
Builds per-user spending profile features and recurring-payment features
that feed both the anomaly model and the subscription detector.
"""

import pandas as pd
import numpy as np


def build_user_spending_features(df):
    df = df.sort_values(["user_id", "timestamp"]).copy()

    # cyclic hour encoding — theoretically sound (hour 23 and hour 0 are
    # adjacent in real time, raw hour_of_day treats them as maximally
    # far apart) and worth trying based on independent notebook
    # experiments on the real dataset. TESTED against this project's
    # labeled synthetic data and it did NOT improve results (ROC-AUC
    # dropped slightly on both the global and personalized models) —
    # not used in FEATURE_COLS by default, kept here as available
    # columns for anyone who wants to experiment further.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    user_stats = df.groupby("user_id")["amount"].agg(
        average_spending="mean",
        median_spending="median",
        spending_std="std",
    ).reset_index()
    df = df.merge(user_stats, on="user_id", how="left")
    df["spending_std"] = df["spending_std"].fillna(0)

    df["days_since_last_transaction"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds() / 86400
    ).fillna(0)

    # rolling 7/30-day spend per user
    df = df.set_index("timestamp")
    roll_frames = []
    for uid, g in df.groupby("user_id"):
        g = g.sort_index()
        g["7_day_spending"] = g["amount"].rolling("7D").sum()
        g["30_day_spending"] = g["amount"].rolling("30D").sum()
        roll_frames.append(g)
    df = pd.concat(roll_frames).reset_index()

    # transaction / merchant / category frequency counts
    df["transaction_frequency"] = df.groupby("user_id")["transaction_id"].transform("count")
    df["merchant_frequency"] = df.groupby(["user_id", "merchant"])["transaction_id"].transform("count")
    df["category_frequency"] = df.groupby(["user_id", "category"])["transaction_id"].transform("count")

    # how far this transaction deviates from the user's own average
    df["amount_deviation"] = (df["amount"] - df["average_spending"]) / df["average_spending"].replace(0, np.nan)
    df["amount_deviation"] = df["amount_deviation"].fillna(0)

    return df


def build_recurring_features(df):
    """Per (user, merchant) pair: interval regularity and amount consistency —
    the signals the subscription detector relies on."""
    df = df.sort_values(["user_id", "merchant", "timestamp"]).copy()
    df["prev_ts"] = df.groupby(["user_id", "merchant"])["timestamp"].shift(1)
    df["days_between_payments"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 86400

    grp = df.groupby(["user_id", "merchant"])
    df["merchant_repeat_count"] = grp["transaction_id"].transform("count")
    df["amount_consistency"] = grp["amount"].transform(lambda s: 1 - (s.std() / s.mean()) if s.mean() else 0)
    df["amount_consistency"] = df["amount_consistency"].fillna(0).clip(lower=0)

    monthly = grp["days_between_payments"].transform("mean")
    df["recurring_interval"] = monthly
    df["monthly_frequency"] = grp["timestamp"].transform(
        lambda s: len(s) / max((s.max() - s.min()).days / 30, 1)
    )

    df = df.drop(columns=["prev_ts"])
    return df


def run(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = build_user_spending_features(df)
    df = build_recurring_features(df)
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset: {df.shape} -> {output_path}")
    return df


if __name__ == "__main__":
    run(
        input_path="data/processed/transactions_clean.csv",
        output_path="data/processed/transactions_featured.csv",
    )
