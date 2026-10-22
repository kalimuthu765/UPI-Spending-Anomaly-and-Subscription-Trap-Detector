"""
Phase 6 — Personalized Model (the project's main MLOps innovation)
A single global threshold treats a Rs.300-median user and a Rs.5,000-median
user identically, which is why the Phase-4 global Isolation Forest scored
weak precision/recall despite decent ROC-AUC. Fix:

    Global Model  +  Per-User Spending Profile  ->  Personalized Detection

Users with enough history (>= MIN_USER_TXNS) get their OWN Isolation
Forest, fit only on their own transactions, so "unusual" is relative to
THEM. Users below that threshold (new users / cold start) fall back to
the global model from anomaly_model.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from anomaly_model import FEATURE_COLS, load_features, explain_row, MODEL_PATH, SCALER_PATH

MIN_USER_TXNS = 20          # below this, use the global model (cold start)
PERSONAL_CONTAMINATION = 0.02
ARTIFACT_DIR = "ml/artifacts"


def score_with_global(df):
    """Reuse the Phase-4 global model for cold-start users."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(df[FEATURE_COLS].fillna(0).values)
    raw = model.decision_function(X)
    score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    pred = (model.predict(X) == -1).astype(int)
    return score, pred


def score_with_personal_model(user_df):
    """Fit a dedicated Isolation Forest on one user's own transaction history."""
    X = user_df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=100, contamination=PERSONAL_CONTAMINATION,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_scaled)
    raw = model.decision_function(X_scaled)
    score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    pred = (model.predict(X_scaled) == -1).astype(int)
    return score, pred


def run_personalized(df):
    df = df.copy()
    df["anomaly_score_personal"] = np.nan
    df["anomaly_prediction_personal"] = np.nan
    df["model_used"] = "global"

    counts = df["user_id"].value_counts()
    eligible_users = counts[counts >= MIN_USER_TXNS].index

    print(f"{len(eligible_users)} / {df['user_id'].nunique()} users have >= {MIN_USER_TXNS} "
          f"transactions and get a personal model; the rest fall back to global.")

    # personal models for eligible users
    for uid in eligible_users:
        mask = df["user_id"] == uid
        score, pred = score_with_personal_model(df.loc[mask])
        df.loc[mask, "anomaly_score_personal"] = score
        df.loc[mask, "anomaly_prediction_personal"] = pred
        df.loc[mask, "model_used"] = "personal"

    # global fallback for the cold-start remainder
    cold_mask = df["model_used"] == "global"
    if cold_mask.any():
        g_score, g_pred = score_with_global(df)
        df.loc[cold_mask, "anomaly_score_personal"] = g_score[cold_mask.values]
        df.loc[cold_mask, "anomaly_prediction_personal"] = g_pred[cold_mask.values]

    df["anomaly_prediction_personal"] = df["anomaly_prediction_personal"].astype(int)
    df["explanation"] = df.apply(explain_row, axis=1)
    return df


def evaluate(df, score_col, pred_col):
    y_true = df["anomaly_label"]
    return {
        "precision": round(precision_score(y_true, df[pred_col], zero_division=0), 4),
        "recall": round(recall_score(y_true, df[pred_col], zero_division=0), 4),
        "f1": round(f1_score(y_true, df[pred_col], zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, df[score_col]), 4),
    }


def run(input_path, output_path):
    df = load_features(input_path)
    df = run_personalized(df)

    metrics = evaluate(df, "anomaly_score_personal", "anomaly_prediction_personal")
    print("\nPersonalized model evaluation vs. injected ground truth:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")

    out_cols = ["transaction_id", "user_id", "timestamp", "amount", "merchant", "category",
                "model_used", "anomaly_score_personal", "anomaly_prediction_personal",
                "anomaly_label", "explanation"]
    df[out_cols].to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    return df, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "ml")
    run(
        input_path="data/processed/transactions_featured.csv",
        output_path="data/processed/anomaly_scored_personalized.csv",
    )
