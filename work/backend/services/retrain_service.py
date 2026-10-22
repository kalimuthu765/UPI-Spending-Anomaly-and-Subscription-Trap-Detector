"""
Phase 12 — Retraining Pipeline (per-user, drift-aware)
Wired into POST /upload/history. On every upload:

    New Transactions -> Merge into stored history -> Drift Detection
        -> Retrain Trigger? -> [yes] Train + Register new ModelVersion
                             -> [no]  Reuse the saved personal model

Retrain triggers on EITHER:
  - no personal model exists yet for this user (first time crossing
    MIN_TXNS_FOR_PERSONAL_MODEL), or
  - significant drift between the baseline the saved model was trained
    on and the transactions added since, or
  - enough new transactions have accumulated since the last train that
    it's worth refreshing even without strong drift (a floor, so a
    model never goes too stale)

Deliberately transaction-history-based only — no time-based trigger.
A user who hasn't uploaded anything new has nothing new to learn from,
so there's nothing to retrain on regardless of how many days passed.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from ..database import models

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml")
sys.path.insert(0, ML_DIR)
from drift_detection import assess_drift, assess_sustained_shift, latest_window_drifted   # noqa: E402
from anomaly_model import FEATURE_COLS     # noqa: E402

MIN_TXNS_FOR_PERSONAL_MODEL = 20
MAX_TXNS_SINCE_RETRAIN_FLOOR = 30   # force a refresh after this many new txns regardless of drift

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ml", "artifacts", "user_models",
)
os.makedirs(MODEL_DIR, exist_ok=True)


def _baseline_path(user_id):
    return os.path.join(MODEL_DIR, f"user_{user_id}_baseline.npy")


def _model_paths(user_id):
    return (
        os.path.join(MODEL_DIR, f"user_{user_id}_isoforest.joblib"),
        os.path.join(MODEL_DIR, f"user_{user_id}_scaler.joblib"),
    )


MAX_TRAIN_TXNS = 500      # train on the most recent N transactions, so old life-phases fade
OUTLIER_MOD_Z = 3.5       # modified z-score (on log amount) above which a txn is a one-off


def _robust_training_rows(df):
    """Drop one-off outliers (a heavy-spending month, a single big purchase)
    from the TRAINING set so the model never learns them as 'normal'. They
    are still scored — and flagged — as anomalies. Uses median/MAD on log
    amounts, which a minority of spikes can't distort."""
    df = df.tail(MAX_TRAIN_TXNS)
    la = np.log1p(df["amount"].values.astype(float))
    med = np.median(la)
    mad = np.median(np.abs(la - med))
    if mad == 0:
        return df
    mz = 0.6745 * (la - med) / mad
    kept = df[np.abs(mz) <= OUTLIER_MOD_Z]
    return kept if len(kept) >= MIN_TXNS_FOR_PERSONAL_MODEL else df


def _train_personal_model(full_df):
    full_df = _robust_training_rows(full_df)
    X = full_df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    return model, scaler, full_df['amount'].values


def decide_and_maybe_retrain(db: Session, user_id: int, full_df):
    """
    Returns (model, scaler, retrained: bool, drift_report: dict|None).
    model/scaler are None if there isn't enough history yet for a
    personal model at all (caller falls back to the global model).
    """
    n = len(full_df)
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    model_path, scaler_path = _model_paths(user_id)

    if n < MIN_TXNS_FOR_PERSONAL_MODEL:
        return None, None, False, None  # not enough history for personal model yet, no drift check

    trained_on_count = profile.trained_on_count if profile and profile.trained_on_count else 0
    has_saved_model = os.path.exists(model_path) and trained_on_count > 0

    drift_report = None
    retrain = False
    retrain_reason = None
    train_df = full_df

    if not has_saved_model:
        retrain = True  # first time crossing the threshold
        retrain_reason = "first_personal_model"
    else:
        full_df_sorted = full_df.sort_values("timestamp").reset_index(drop=True)
        # baseline = exactly the amounts the saved model was trained on (a small
        # sidecar file), so after a lifestyle change the next drift check
        # compares against the NEW regime, not a mix of old and new
        bp = _baseline_path(user_id)
        baseline_amounts = (np.load(bp) if os.path.exists(bp)
                            else full_df_sorted["amount"].iloc[:trained_on_count].values)
        current_amounts = full_df_sorted["amount"].iloc[trained_on_count:].values
        n_new = len(current_amounts)

        if n_new == 0:
            # nothing new since last train — reuse without even checking drift
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, False, None

        drift_report = assess_drift(baseline_amounts, current_amounts)
        drift_report["shift_type"] = "none"
        latest_drifted = latest_window_drifted(baseline_amounts, current_amounts)
        floor_due = n_new >= MAX_TXNS_SINCE_RETRAIN_FLOOR
        if drift_report["drift_detected"] and assess_sustained_shift(baseline_amounts, current_amounts):
            # a real, continuing change (college -> job): adapt, and learn
            # from the new regime only — the old regime no longer applies
            retrain, retrain_reason = True, "sustained_shift"
            drift_report["shift_type"] = "sustained"
            train_df = full_df_sorted.iloc[trained_on_count:]
        elif drift_report["drift_detected"] and latest_drifted:
            # could be a one-off heavy month still in progress: keep the current
            # model so the spike is flagged as anomalies, and wait to see if it continues
            drift_report["shift_type"] = "temporary_spike_suspected"
        else:
            if drift_report["drift_detected"]:
                drift_report["shift_type"] = "spike_resolved"   # spending went back to normal
            if floor_due:
                # routine refresh; the robust training filter keeps any past spike out
                retrain, retrain_reason = True, "count_floor"

    if retrain:
        model, scaler, trained_amounts = _train_personal_model(train_df)
        np.save(_baseline_path(user_id), trained_amounts)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        if profile is None:
            profile = models.UserProfile(user_id=user_id)
            db.add(profile)
        profile.has_personal_model = True
        profile.personal_model_path = model_path
        profile.trained_on_count = n
        db.commit()

        version_count = db.query(models.ModelVersion).filter(
            models.ModelVersion.user_id == user_id
        ).count()
        db.add(models.ModelVersion(
            user_id=user_id, name="personal_isolation_forest", version=f"v{version_count + 1}",
            scope="personalized", trained_on_count=n,
        ))
        db.commit()
    else:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    if drift_report is not None:
        drift_report["retrain_reason"] = retrain_reason
        db.add(models.DriftReport(
            user_id=user_id, psi=drift_report["psi"], ks_statistic=drift_report["ks_statistic"],
            ks_pvalue=drift_report["ks_pvalue"], baseline_mean=drift_report["baseline_mean"],
            current_mean=drift_report["current_mean"], severity=drift_report["severity"],
            drift_detected=drift_report["drift_detected"], retrain_triggered=retrain,
        ))
        db.commit()

    return model, scaler, retrain, drift_report


