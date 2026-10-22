"""
Phase 2 — Data Preprocessing
Cleans the raw/synthetic transaction data: missing values, duplicates,
date parsing, impossible-value filtering. Outputs data/processed/transactions_clean.csv
"""

import pandas as pd
import numpy as np


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    df = df.copy()
    if df["amount"].isna().any():
        df["amount"] = df["amount"].fillna(df["amount"].median())
    if df["category"].isna().any():
        df["category"] = df["category"].fillna(df["category"].mode()[0])
    df["merchant"] = df["merchant"].fillna("unknown")
    df["subscription_id"] = df["subscription_id"].fillna("")
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"])
    print(f"Removed {before - len(df):,} duplicate transaction_id rows")
    return df


def validate_transactions(df):
    before = len(df)
    df = df[df["amount"] > 0]
    df = df[df["timestamp"].notna()]
    print(f"Removed {before - len(df):,} invalid rows (amount<=0 or bad timestamp)")
    return df


def extract_date_features(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def run(input_path, output_path):
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_transactions(df)
    df = extract_date_features(df)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"\nClean dataset: {len(df):,} rows -> {output_path}")
    return df


if __name__ == "__main__":
    run(
        input_path="data/synthetic/final_transactions.csv",
        output_path="data/processed/transactions_clean.csv",
    )
