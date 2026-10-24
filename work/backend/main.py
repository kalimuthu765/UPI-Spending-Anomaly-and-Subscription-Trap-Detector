"""
FastAPI Backend — Phase 8 + Phase 9 (auth + persistence)

Two upload flows:
  POST /upload/quick-analysis  — anonymous, one-off, nothing saved
                                  (the original flow, still useful for a
                                  quick check without an account)
  POST /upload/history         — requires login. New transactions are
                                  MERGED into the user's stored history,
                                  and the personal model is retrained on
                                  the FULL accumulated set — so a college
                                  student's profile and an employee's
                                  profile each keep improving with every
                                  upload, independently, tied to their
                                  own login.

Run with:  uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.services.analysis_service import analyze_upload, parse_and_validate, build_features, analyze_df   # noqa: E402
from backend.services import persist_service, retrain_service                                    # noqa: E402
from backend.database.db import get_db, init_db                                                  # noqa: E402
from backend.database import models                                                               # noqa: E402
from backend.auth import (                                                                        # noqa: E402
    hash_password, verify_password, create_access_token, get_current_user,
)

app = FastAPI(
    title="UPI Anomaly & Subscription Detector API",
    description="Login, upload UPI transaction history, and get anomalies, "
                 "recurring subscriptions, and a spending breakdown — with "
                 "your profile persisting and improving across uploads.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class StripApiPrefixMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.scope.get("path", "")
        if path.startswith("/api/"):
            request.scope["path"] = path[4:]
        elif path == "/api":
            request.scope["path"] = "/api-info"
        return await call_next(request)

app.add_middleware(StripApiPrefixMiddleware)


@app.on_event("startup")
def on_startup():
    init_db()


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ---------- auth ----------

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    display_name: str = None


@app.post("/auth/register")
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == payload.email).first():
        raise HTTPException(400, "An account with this email already exists.")
    user = models.User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        display_name=payload.display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password.")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}


@app.get("/auth/me")
def read_me(current_user: models.User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email, "display_name": current_user.display_name}


# ---------- the core, login-tied, accumulating upload flow ----------

@app.post("/upload/history")
async def upload_history(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload transaction history while logged in — CSV or PDF (Google Pay,
    PhonePe, or a generic bank statement). New rows are merged into
    everything you've previously uploaded, and the anomaly/subscription
    analysis re-runs on your FULL accumulated history — so your profile
    keeps improving with every upload instead of resetting.
    """
    if not file.filename.lower().endswith((".csv", ".pdf", ".xlsx", ".xls", ".json", ".docx", ".doc")):
        raise HTTPException(400, "Please upload a .csv, .xlsx/.xls, .json, .docx, or .pdf file.")
    contents = await file.read()
    try:
        new_df = parse_and_validate(contents, user_id=str(current_user.id), filename=file.filename)
        full_df, new_rows = persist_service.merge_new_upload(db, current_user.id, new_df)
        new_credits = persist_service.save_credits(db, current_user.id, new_df.attrs.get("credits"))
        featured_df = build_features(full_df)
        model, scaler, retrained, drift_report = retrain_service.decide_and_maybe_retrain(
            db, current_user.id, featured_df
        )
        result = analyze_df(featured_df, user_id=str(current_user.id), reuse_model=model, reuse_scaler=scaler)
        if retrained:
            reason = (drift_report or {}).get("retrain_reason") or "first_personal_model"
            label = {"first_personal_model": "personal (first model trained on your history)",
                     "sustained_shift": "personal (retrained: your spending pattern changed and stayed changed)",
                     "count_floor": "personal (retrained: enough new transactions)"}.get(reason, "personal (retrained)")
            result["summary"]["model_used"] = label
            for a in result["anomalies"]:
                a["model_used"] = label
        if drift_report and drift_report.get("shift_type") == "temporary_spike_suspected":
            result["insights"].insert(0, "Your recent spending is well above your usual pattern. We're treating it as a "
                                         "one-off for now, so it's flagged as unusual and your normal baseline is unchanged. "
                                         "If it continues, your model will adapt automatically.")
        persist_service.save_analysis_results(db, current_user.id, result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["new_transactions_added"] = new_rows
    result["new_credits_added"] = new_credits
    result["model_retrained_this_upload"] = retrained
    result["drift_report"] = drift_report
    return result


@app.get("/me/drift-history")
def my_drift_history(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Every drift check and model version logged for the logged-in user —
    the Phase 11/12 audit trail."""
    reports = db.query(models.DriftReport).filter(
        models.DriftReport.user_id == current_user.id
    ).order_by(models.DriftReport.created_at.desc()).all()
    versions = db.query(models.ModelVersion).filter(
        models.ModelVersion.user_id == current_user.id
    ).order_by(models.ModelVersion.registered_at.desc()).all()
    return {
        "drift_reports": [{
            "psi": r.psi, "ks_statistic": r.ks_statistic, "ks_pvalue": r.ks_pvalue,
            "baseline_mean": r.baseline_mean, "current_mean": r.current_mean,
            "severity": r.severity, "drift_detected": r.drift_detected,
            "retrain_triggered": r.retrain_triggered, "created_at": str(r.created_at),
        } for r in reports],
        "model_versions": [{
            "version": v.version, "scope": v.scope, "trained_on_count": v.trained_on_count,
            "registered_at": str(v.registered_at),
        } for v in versions],
    }


@app.get("/me/dashboard")
def my_dashboard(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Read the persisted, accumulated dashboard for the logged-in user —
    no re-upload needed."""
    dashboard = persist_service.get_dashboard(db, current_user.id)
    if dashboard is None:
        raise HTTPException(404, "No history uploaded yet — POST a CSV to /upload/history first.")
    return dashboard


@app.get("/me/transactions")
def my_transactions(limit: int = 50, current_user: models.User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    """Transaction History panel — most recent transactions for the
    logged-in user, newest first."""
    return persist_service.get_recent_transactions(db, current_user.id, limit=limit)


@app.get("/me/insights")
def my_insights(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Just the natural-language insights + period analysis, for a
    lighter call than the full dashboard."""
    dashboard = persist_service.get_dashboard(db, current_user.id)
    if dashboard is None:
        raise HTTPException(404, "No history uploaded yet.")
    return {"period_analysis": dashboard["period_analysis"], "insights": dashboard["insights"]}


@app.get("/me/model-status")
def my_model_status(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    dashboard = persist_service.get_dashboard(db, current_user.id)
    if dashboard is None:
        raise HTTPException(404, "No history uploaded yet.")
    return dashboard["model_status"]


@app.get("/me/drift-status")
def my_drift_status(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    dashboard = persist_service.get_dashboard(db, current_user.id)
    if dashboard is None:
        raise HTTPException(404, "No history uploaded yet.")
    return dashboard["drift_status"]


@app.post("/me/retrain")
def my_manual_retrain(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Manual retrain trigger (spec's POST /model/retrain) — forces a
    fresh personal model on the user's full stored history right now,
    bypassing the drift check (useful right after a big upload, or for
    a demo)."""
    txns = db.query(models.Transaction).filter(models.Transaction.user_id == current_user.id).all()
    if len(txns) < retrain_service.MIN_TXNS_FOR_PERSONAL_MODEL:
        raise HTTPException(
            400, f"Need at least {retrain_service.MIN_TXNS_FOR_PERSONAL_MODEL} transactions "
                 f"for a personal model (have {len(txns)})."
        )
    full_df = pd.DataFrame([{
        "transaction_id": t.transaction_id, "user_id": str(current_user.id), "timestamp": t.timestamp,
        "amount": t.amount, "merchant": t.merchant, "category": t.category,
        "payment_mode": t.payment_mode, "recurring_flag": 0, "subscription_id": "", "anomaly_label": 0,
    } for t in txns])
    featured_df = build_features(full_df)

    model, scaler = retrain_service._train_personal_model(featured_df)
    model_path, scaler_path = retrain_service._model_paths(current_user.id)
    import joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == current_user.id).first()
    if profile is None:
        profile = models.UserProfile(user_id=current_user.id)
        db.add(profile)
    profile.has_personal_model = True
    profile.personal_model_path = model_path
    profile.trained_on_count = len(full_df)
    db.commit()

    version_count = db.query(models.ModelVersion).filter(models.ModelVersion.user_id == current_user.id).count()
    db.add(models.ModelVersion(
        user_id=current_user.id, name="personal_isolation_forest", version=f"v{version_count + 1}",
        scope="personalized", trained_on_count=len(full_df),
    ))
    db.commit()

    return {"retrained": True, "trained_on_count": len(full_df), "version": f"v{version_count + 1}"}


# ---------- anonymous, one-off flow (no login, nothing saved) ----------

@app.post("/upload/quick-analysis")
async def upload_quick_analysis(file: UploadFile = File(...), user_id: str = "uploaded_user"):
    """One-off analysis with no account and nothing persisted — useful for
    a quick check without signing up. CSV or PDF."""
    if not file.filename.lower().endswith((".csv", ".pdf", ".xlsx", ".xls", ".json", ".docx", ".doc")):
        raise HTTPException(400, "Please upload a .csv, .xlsx/.xls, .json, .docx, or .pdf file.")
    contents = await file.read()
    try:
        result = analyze_upload(contents, user_id=user_id, filename=file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


# ---------- single-transaction prediction ----------

class TransactionIn(BaseModel):
    user_id: str
    amount: float
    merchant: str
    category: str = "Other"


@app.post("/predict/anomaly")
def predict_anomaly(txn: TransactionIn):
    import joblib
    import numpy as np
    model_path = os.path.join(PROJECT_ROOT, "ml", "artifacts", "isolation_forest.joblib")
    scaler_path = model_path.replace("isolation_forest", "scaler")
    if not os.path.exists(model_path):
        raise HTTPException(503, "Model not trained yet — run ml/anomaly_model.py first.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    row = np.array([[txn.amount, 0, 1, 1, txn.amount, txn.amount, 12, 1]])
    X = scaler.transform(row)
    score = model.decision_function(X)[0]
    pred = model.predict(X)[0] == -1
    return {
        "anomaly": bool(pred),
        "score": round(float(1 - (score + 0.5)), 4),
        "reason": "Amount and context evaluated against the global spending model" if pred
                  else "Within normal range for the global model",
    }


# ---------- read endpoints over the pre-processed synthetic dataset ----------

@app.get("/transactions/{user_id}")
def get_transactions(user_id: str, limit: int = 50):
    path = os.path.join(DATA_DIR, "transactions_featured.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "Dataset not generated yet.")
    df = pd.read_csv(path)
    rows = df[df["user_id"] == user_id].sort_values("timestamp", ascending=False).head(limit)
    if rows.empty:
        raise HTTPException(404, f"No transactions found for {user_id}")
    return rows.to_dict(orient="records")


@app.get("/dashboard/{user_id}")
def get_dashboard(user_id: str):
    path = os.path.join(DATA_DIR, "transactions_featured.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "Dataset not generated yet.")
    df = pd.read_csv(path)
    user_df = df[df["user_id"] == user_id]
    if user_df.empty:
        raise HTTPException(404, f"No data for {user_id}")
    return {
        "total_spending": round(float(user_df["amount"].sum()), 2),
        "monthly_spending": round(float(user_df.groupby("month")["amount"].sum().mean()), 2),
        "anomalies": int(user_df["anomaly_label"].sum()),
        "spending_by_category": user_df.groupby("category")["amount"].sum().round(2).to_dict(),
    }


@app.get("/subscriptions/{user_id}")
def get_subscriptions(user_id: str):
    path = os.path.join(DATA_DIR, "subscriptions_detected.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "Run ml/subscription_model.py first.")
    df = pd.read_csv(path)
    return df[df["user_id"] == user_id].to_dict(orient="records")


@app.get("/alerts/{user_id}")
def get_alerts(user_id: str):
    path = os.path.join(DATA_DIR, "anomaly_scored_personalized.csv")
    if not os.path.exists(path):
        raise HTTPException(503, "Run ml/personalized_model.py first.")
    df = pd.read_csv(path)
    rows = df[(df["user_id"] == user_id) & (df["anomaly_prediction_personal"] == 1)]
    return rows.to_dict(orient="records")


@app.get("/model/status")
def model_status():
    model_path = os.path.join(PROJECT_ROOT, "ml", "artifacts", "isolation_forest.joblib")
    return {
        "global_model_trained": os.path.exists(model_path),
        "model_path": model_path if os.path.exists(model_path) else None,
    }


@app.get("/api-info")
def api_info():
    return {
        "message": "UPI Anomaly & Subscription Detector API",
        "flow": "POST /auth/register -> POST /auth/login -> POST /upload/history (with Bearer token) "
                "-> GET /me/dashboard",
        "no_login_flow": "POST /upload/quick-analysis (no account, nothing saved)",
        "docs": "/docs",
    }


FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")


@app.get("/")
def root():
    index_file = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.exists(index_file):
        from starlette.responses import FileResponse
        return FileResponse(index_file)
    return api_info()


if os.path.exists(FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(404)
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Not found"}
