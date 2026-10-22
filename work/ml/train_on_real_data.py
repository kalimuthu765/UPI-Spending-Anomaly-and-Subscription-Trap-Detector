"""
Train Directly on the Real Dataset (upi_transactions_2024.csv)
-----------------------------------------------------------------
Unlike the synthetic pipeline, this dataset has:
  - a REAL fraud_flag label (not a simulated one) -> supervised learning
    is possible and will beat unsupervised Isolation Forest here
  - NO user_id and NO merchant name -> per-user personalization (Phase 6)
    and merchant-level features aren't possible on this data as-is; this
    trains a GLOBAL fraud classifier over transaction-level signals only
    (amount, category, device, network, time-of-day, bank pairing, etc.)

Two models are trained and compared:
  1. Isolation Forest (unsupervised) — same family as the main pipeline,
     contamination set to the REAL fraud rate (~0.19%)
  2. Random Forest (supervised, class-weighted) — uses fraud_flag
     directly; expected to substantially outperform (1), since with real
     imbalance this extreme (0.19%), unsupervised anomaly detection has
     very little to key off without ground truth.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report

RAW_PATH = "data/raw/upi_transactions_2024.csv"
ARTIFACT_DIR = "ml/artifacts"

NUMERIC_FEATURES = ["amount (INR)", "hour_of_day", "is_weekend"]
CATEGORICAL_FEATURES = [
    "transaction type", "merchant_category", "sender_age_group", "receiver_age_group",
    "sender_bank", "receiver_bank", "device_type", "network_type", "day_of_week",
]


def load_and_prepare(path):
    df = pd.read_csv(path)
    df["same_bank_transfer"] = (df["sender_bank"] == df["receiver_bank"]).astype(int)
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=20), CATEGORICAL_FEATURES),
    ])


def train_isolation_forest(X_train, y_train, preprocessor, real_fraud_rate):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", IsolationForest(
            n_estimators=200, contamination=real_fraud_rate, random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(X_train)  # unsupervised: doesn't see y_train
    return pipe


def train_random_forest(X_train, y_train, preprocessor):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_isolation_forest(pipe, X_test, y_test):
    raw = pipe.named_steps["model"].decision_function(pipe.named_steps["prep"].transform(X_test))
    score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    pred = (pipe.named_steps["model"].predict(pipe.named_steps["prep"].transform(X_test)) == -1).astype(int)
    return {
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, score), 4),
    }, pred, score


def evaluate_random_forest(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    return {
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }, pred, proba


def main():
    print("Loading real dataset (250k rows)...")
    df = load_and_prepare(RAW_PATH)
    real_fraud_rate = df["fraud_flag"].mean()
    print(f"Real fraud rate: {real_fraud_rate*100:.3f}% ({df['fraud_flag'].sum()} of {len(df)})")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["fraud_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n--- Model 1: Isolation Forest (unsupervised) ---")
    if_pipe = train_isolation_forest(X_train, y_train, build_preprocessor(), real_fraud_rate)
    if_metrics, _, _ = evaluate_isolation_forest(if_pipe, X_test, y_test)
    for k, v in if_metrics.items():
        print(f"  {k:10s}: {v}")

    print("\n--- Model 2: Random Forest (supervised, class-weighted) ---")
    rf_pipe = train_random_forest(X_train, y_train, build_preprocessor())
    rf_metrics, rf_pred, rf_proba = evaluate_random_forest(rf_pipe, X_test, y_test)
    for k, v in rf_metrics.items():
        print(f"  {k:10s}: {v}")
    print("\nFull classification report (Random Forest):")
    print(classification_report(y_test, rf_pred, target_names=["legit", "fraud"], zero_division=0))

    print("=" * 55)
    print(f"{'metric':10s} {'IsolationForest':>16s} {'RandomForest':>14s}")
    for k in if_metrics:
        print(f"{k:10s} {if_metrics[k]:>16.4f} {rf_metrics[k]:>14.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(rf_pipe, os.path.join(ARTIFACT_DIR, "real_data_fraud_rf.joblib"))
    joblib.dump(if_pipe, os.path.join(ARTIFACT_DIR, "real_data_fraud_isoforest.joblib"))
    print(f"\nBoth models saved to {ARTIFACT_DIR}/")
    print("\nNote: this dataset has no user_id or merchant name, so per-user "
          "personalization (Phase 6) and merchant-level features don't apply here — "
          "these are GLOBAL models over transaction-level signals only.")

    return if_metrics, rf_metrics


if __name__ == "__main__":
    main()
