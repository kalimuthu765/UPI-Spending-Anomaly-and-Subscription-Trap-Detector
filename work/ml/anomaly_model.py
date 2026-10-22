"""
Phase 4 — Anomaly Detection (Isolation Forest)
Trains on the engineered features, evaluates against the injected
anomaly_label ground truth, and generates human-readable explanations
("Amount is 15.6x higher than your typical transaction...") instead of
a bare "fraud detected" flag — per the spec.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

FEATURE_COLS = [
    "amount", "amount_deviation", "transaction_frequency", "merchant_frequency",
    "7_day_spending", "30_day_spending", "hour", "category_frequency",
]

MODEL_PATH = "ml/artifacts/isolation_forest.joblib"
SCALER_PATH = "ml/artifacts/scaler.joblib"


def load_features(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for c in FEATURE_COLS:
        df[c] = df[c].fillna(0)
    return df


def train(df, contamination=0.015):
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler, X_scaled


def score(df, model, scaler, X_scaled):
    df = df.copy()
    raw_scores = model.decision_function(X_scaled)          # higher = more normal
    df["anomaly_score"] = (1 - (raw_scores - raw_scores.min()) /
                            (raw_scores.max() - raw_scores.min()))  # 0=normal, 1=anomalous
    df["anomaly_prediction"] = (model.predict(X_scaled) == -1).astype(int)
    return df


def explain_row(row):
    """Human-readable reason, not a bare 'fraud detected' flag."""
    reasons = []
    base = row.get("typical_amount", row["average_spending"])   # median-based when available: spikes can't inflate it
    multiple = row["amount"] / max(base, 1)
    if abs(row["amount_deviation"]) > 1.0 or multiple >= 3:
        reasons.append(f"Amount is {multiple:.1f}x higher than your typical transaction")
    if row["merchant_frequency"] <= 1:
        reasons.append("Merchant has not appeared in your recent history")
    # hour 0 usually means the statement gave only a date (no time), so it says nothing about the hour
    if 1 <= row["hour"] < 5:
        reasons.append(f"Unusual transaction hour ({int(row['hour'])}:00)")
    if not reasons:
        reasons.append("Combination of spending pattern signals deviates from your usual pattern")
    return "; ".join(reasons)


def evaluate(df):
    y_true = df["anomaly_label"]
    y_pred = df["anomaly_prediction"]
    metrics = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, df["anomaly_score"]), 4),
    }
    return metrics


def run(input_path, output_path):
    import os
    os.makedirs("ml/artifacts", exist_ok=True)

    df = load_features(input_path)
    model, scaler, X_scaled = train(df)
    df = score(df, model, scaler, X_scaled)
    df["explanation"] = df.apply(explain_row, axis=1)

    metrics = evaluate(df)
    print("Isolation Forest evaluation vs. injected ground truth:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    out_cols = ["transaction_id", "user_id", "timestamp", "amount", "merchant", "category",
                "anomaly_score", "anomaly_prediction", "anomaly_label", "explanation"]
    df[out_cols].to_csv(output_path, index=False)
    print(f"Scored transactions -> {output_path}")

    print("\nSample flagged anomalies:")
    sample = df[df["anomaly_prediction"] == 1].sort_values("anomaly_score", ascending=False).head(5)
    for _, r in sample.iterrows():
        print(f"  [{r['user_id']}] Rs.{r['amount']:.0f} at {r['merchant']} "
              f"(score={r['anomaly_score']:.2f}) -> {r['explanation']}")

    return df, metrics


if __name__ == "__main__":
    run(
        input_path="data/processed/transactions_featured.csv",
        output_path="data/processed/anomaly_scored.csv",
    )
