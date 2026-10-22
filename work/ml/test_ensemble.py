"""
Ensemble test — combines Isolation Forest, Local Outlier Factor, and HBOS
(the techniques from independent notebook experimentation on the real
dataset) via majority vote, and compares against the single-model
baseline on THIS project's labeled synthetic data — same evaluation
harness as anomaly_model.py, for a fair before/after comparison.
"""
import sys
sys.path.insert(0, "ml")
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pyod.models.hbos import HBOS

from anomaly_model import FEATURE_COLS, load_features

df = load_features("data/processed/transactions_featured.csv")
X = df[FEATURE_COLS].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
contamination = 0.015

print("Training Isolation Forest...")
iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
iso_pred = (iso.fit_predict(X_scaled) == -1).astype(int)

print("Training LOF...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
lof_pred = (lof.fit_predict(X_scaled) == -1).astype(int)

print("Training HBOS...")
hbos = HBOS(contamination=contamination)
hbos.fit(X_scaled)
hbos_pred = hbos.predict(X_scaled)

votes = iso_pred + lof_pred + hbos_pred
y_true = df["anomaly_label"].values


def eval_pred(name, pred):
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    print(f"{name:30s} precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")
    return f1


print("\n--- Individual models ---")
eval_pred("Isolation Forest (solo)", iso_pred)
eval_pred("LOF (solo)", lof_pred)
eval_pred("HBOS (solo)", hbos_pred)

print("\n--- Ensemble (vote threshold) ---")
for threshold in [1, 2, 3]:
    ensemble_pred = (votes >= threshold).astype(int)
    eval_pred(f"Ensemble (>={threshold}/3 models agree)", ensemble_pred)

print("\nVote distribution:")
print(pd.Series(votes).value_counts().sort_index())

print("""
CONCLUSION: Isolation Forest alone (F1 0.1653) beats every ensemble
threshold tested (best ensemble F1: 0.1448 at ">=1 model agrees").
LOF and HBOS individually underperform Isolation Forest on this data,
so combining them dilutes rather than strengthens the signal. Requiring
more model agreement (>=2, >=3) raises precision but recall collapses,
netting a worse F1 either way.

This is a genuine negative result, not a bug — kept in the codebase as
a documented experiment: two techniques suggested by independent
notebook research (this file's ensemble, and cyclic hour encoding
tested in anomaly_model.py's git history) were both properly tested
against labeled data and neither improved on the existing baseline.
Isolation Forest alone remains the model used in production
(anomaly_model.py / personalized_model.py).
""")

