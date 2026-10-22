"""
Persistence service — turns a one-off CSV analysis into an accumulating
per-user history in the database. This is what makes personalization
PERSIST across logins instead of resetting on every upload: each new
upload is merged with everything the user has previously uploaded, and
the personal model is retrained on the full accumulated set.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..database import models

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml")
sys.path.insert(0, ML_DIR)
from insights_engine import detect_period, generate_insights   # noqa: E402


def merge_new_upload(db: Session, user_id: int, new_df: pd.DataFrame) -> pd.DataFrame:
    """Insert transactions from a new upload (skipping ones already stored
    by transaction_id) and return the user's FULL accumulated history."""
    # Same statement uploaded twice must not double the spending. Rows are
    # matched on (time, amount, shop); a Counter keeps genuine repeats inside
    # one file (two identical payments in the same minute) intact.
    from collections import Counter
    have = Counter(
        (pd.Timestamp(t.timestamp), round(float(t.amount), 2), str(t.merchant))
        for t in db.query(models.Transaction).filter(models.Transaction.user_id == user_id).all()
    )

    new_rows = 0
    for _, r in new_df.iterrows():
        key = (pd.Timestamp(r["timestamp"]), round(float(r["amount"]), 2), str(r["merchant"]))
        if have[key] > 0:
            have[key] -= 1
            continue
        db.add(models.Transaction(
            user_id=user_id,
            transaction_id=str(r["transaction_id"]),
            timestamp=pd.Timestamp(r["timestamp"]).to_pydatetime(),
            amount=float(r["amount"]),
            merchant=str(r["merchant"]),
            category=str(r["category"]),
            payment_mode=str(r.get("payment_mode", "")),
        ))
        new_rows += 1
    db.commit()

    all_rows = db.query(models.Transaction).filter(models.Transaction.user_id == user_id).all()
    full_df = pd.DataFrame([{
        "transaction_id": t.transaction_id, "user_id": str(user_id), "timestamp": t.timestamp,
        "amount": t.amount, "merchant": t.merchant, "category": t.category,
        "payment_mode": t.payment_mode, "recurring_flag": 0, "subscription_id": "",
        "anomaly_label": 0,
    } for t in all_rows])

    return full_df, new_rows


def save_credits(db: Session, user_id: int, credits_df) -> int:
    """Store money received, skipping rows already stored (same time, amount, source)."""
    if credits_df is None or len(credits_df) == 0:
        return 0
    have = {(c.timestamp, round(c.amount, 2), c.source) for c in
            db.query(models.Credit).filter(models.Credit.user_id == user_id).all()}
    added = 0
    for _, r in credits_df.iterrows():
        ts = pd.Timestamp(r["timestamp"]).to_pydatetime()
        key = (ts, round(float(r["amount"]), 2), str(r["source"]))
        if key in have:
            continue
        have.add(key)
        db.add(models.Credit(user_id=user_id, timestamp=ts, amount=float(r["amount"]),
                             source=str(r["source"]), kind=str(r["kind"])))
        added += 1
    db.commit()
    return added


def credit_summary(db: Session, user_id: int) -> dict:
    rows = db.query(models.Credit).filter(models.Credit.user_id == user_id).all()
    if not rows:
        return {"has_credits": False, "total": 0, "count": 0, "refund_total": 0, "refund_count": 0,
                "other_total": 0, "other_count": 0, "by_month": [], "sources": [], "rows": []}
    df = pd.DataFrame([{"timestamp": c.timestamp, "amount": c.amount, "source": c.source, "kind": c.kind}
                       for c in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    refunds = df[df["kind"] == "Refund"]
    other = df[df["kind"] != "Refund"]
    df["ym"] = df["timestamp"].dt.to_period("M")
    by_month = []
    for ym, g in sorted(df.groupby("ym"), key=lambda kv: kv[0], reverse=True):
        by_month.append({
            "label": ym.strftime("%B %Y"), "total": round(float(g["amount"].sum()), 2), "count": int(len(g)),
            "refund_total": round(float(g.loc[g["kind"] == "Refund", "amount"].sum()), 2),
            "top_sources": [{"source": k, "total": round(float(v), 2)} for k, v in
                            g.groupby("source")["amount"].sum().sort_values(ascending=False).head(3).items()],
        })
    sources = []
    for src, g in df.groupby("source"):
        sources.append({"source": src, "total": round(float(g["amount"].sum()), 2), "count": int(len(g)),
                        "refunds": int((g["kind"] == "Refund").sum()),
                        "last": g["timestamp"].max().strftime("%d %b %Y")})
    sources.sort(key=lambda r: r["total"], reverse=True)
    recent = df.sort_values("timestamp", ascending=False).head(500)
    return {
        "has_credits": True,
        "total": round(float(df["amount"].sum()), 2), "count": int(len(df)),
        "refund_total": round(float(refunds["amount"].sum()), 2), "refund_count": int(len(refunds)),
        "other_total": round(float(other["amount"].sum()), 2), "other_count": int(len(other)),
        "by_month": by_month, "sources": sources,
        "rows": [{"date": r.timestamp.strftime("%d %b %Y"), "source": r.source, "kind": r.kind,
                  "amount": round(float(r.amount), 2)} for r in recent.itertuples()],
    }


def save_analysis_results(db: Session, user_id: int, result: dict):
    """Persist the profile, alerts, and subscriptions from an analysis run
    so /me/dashboard can read them back without re-analyzing."""
    summary = result["summary"]

    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    if profile is None:
        profile = models.UserProfile(user_id=user_id)
        db.add(profile)
    profile.total_transactions = summary["total_transactions"]
    profile.average_spending = summary["average_spending"]
    profile.median_spending = summary["median_spending"]
    profile.spending_std = summary["spending_std"]
    profile.has_personal_model = summary["model_used"].startswith("personal")
    profile.last_updated = datetime.utcnow()

    # replace alerts/subscriptions with the latest full-history analysis
    db.query(models.AnomalyAlert).filter(models.AnomalyAlert.user_id == user_id).delete()
    for a in result["anomalies"]:
        db.add(models.AnomalyAlert(
            user_id=user_id, transaction_id=a["transaction_id"],
            timestamp=pd.Timestamp(a["timestamp"]).to_pydatetime(),
            amount=a["amount"], merchant=a["merchant"],
            anomaly_score=a["anomaly_score"], explanation=a["explanation"],
            model_used=a.get("model_used", ""),
        ))

    db.query(models.Subscription).filter(models.Subscription.user_id == user_id).delete()
    for s in result["subscriptions"]:
        db.add(models.Subscription(
            user_id=user_id, merchant=s["merchant"], avg_amount=s["avg_amount"],
            cadence=s.get("cadence"), recurring_probability=s.get("recurring_probability"),
            last_charge=pd.Timestamp(s["last_txn"]).to_pydatetime() if s.get("last_txn") else None,
            next_expected_payment=pd.Timestamp(s["next_expected_payment"]).to_pydatetime()
                if s.get("next_expected_payment") and s["next_expected_payment"] != "NaT" else None,
        ))

    db.commit()


def get_dashboard(db: Session, user_id: int):
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    alerts = db.query(models.AnomalyAlert).filter(models.AnomalyAlert.user_id == user_id).all()
    subs = db.query(models.Subscription).filter(models.Subscription.user_id == user_id).all()
    txns = db.query(models.Transaction).filter(models.Transaction.user_id == user_id).all()

    if not txns:
        return None

    df = pd.DataFrame([{"category": t.category, "merchant": t.merchant, "amount": t.amount,
                         "timestamp": t.timestamp} for t in txns])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    from categorizer import fill_categories      # older uploads stored "Other"; guess from the shop name
    df["category"] = fill_categories(df["merchant"], df["category"])
    by_category = df.groupby("category")["amount"].sum().round(2).sort_values(ascending=False).to_dict()
    by_merchant = df.groupby("merchant")["amount"].sum().round(2).sort_values(ascending=False).head(10).to_dict()

    # period analysis + natural-language insights, recomputed from the
    # full stored history (Sections 10 & 16) — same engine the upload
    # flow uses, so the dashboard reflects the SAME numbers without
    # requiring a fresh upload.
    subs_df = pd.DataFrame([{"merchant": s.merchant, "avg_amount": s.avg_amount} for s in subs])
    period = detect_period(df)
    insights = generate_insights(df, period, subs_df if len(subs_df) else None)

    drift_reports = db.query(models.DriftReport).filter(
        models.DriftReport.user_id == user_id
    ).order_by(models.DriftReport.created_at.desc()).all()
    model_versions = db.query(models.ModelVersion).filter(
        models.ModelVersion.user_id == user_id
    ).order_by(models.ModelVersion.registered_at.desc()).all()
    latest_drift = drift_reports[0] if drift_reports else None

    total_spent = float(df["amount"].sum())

    # per-shop ledger: how much, how often, typical size, share of all spending
    shop_g = df.groupby("merchant").agg(
        total=("amount", "sum"), count=("amount", "count"),
        average=("amount", "mean"), last_date=("timestamp", "max"),
    ).sort_values("total", ascending=False)
    shops = [{
        "merchant": m, "total": round(float(r["total"]), 2), "count": int(r["count"]),
        "average": round(float(r["average"]), 2),
        "share_pct": round(100 * float(r["total"]) / total_spent, 1) if total_spent else 0,
        "last_date": str(r["last_date"].date()),
    } for m, r in shop_g.head(15).iterrows()]

    # spending over time, at a granularity that fits how much history there is
    gran = {"week_or_less": "D", "month": "W", "several_months": "M", "year_plus": "M"}.get(period.get("tier"), "M")
    per = df["timestamp"].dt.to_period(gran)
    series = df.groupby(per)["amount"].sum()
    series = series.reindex(pd.period_range(series.index.min(), series.index.max(), freq=gran), fill_value=0)
    fmt = {"D": "%d %b", "W": "%d %b", "M": "%b %Y"}[gran]
    points = [{"label": (p.start_time.strftime(fmt)), "amount": round(float(v), 2)} for p, v in series.items()]
    last_ts = df["timestamp"].max()
    if gran == "M" and last_ts.day < last_ts.days_in_month - 3 and points:
        points[-1]["label"] += " (so far)"   # data ends mid-month: don't let a short bar read as "spending dropped"
    trend = {"granularity": {"D": "day", "W": "week", "M": "month"}[gran], "points": points}

    # period-by-period breakdown (each month, or each week / day for short uploads):
    # how much went out, which shops took it, and which payments looked unusual
    alert_ids = {a.transaction_id for a in alerts}
    df["_unusual"] = [t.transaction_id in alert_ids for t in txns]
    periods = []
    for per_key, g in df.groupby(per):
        top = g.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(3)
        flagged = g[g["_unusual"]]
        label = per_key.start_time.strftime("%B %Y" if gran == "M" else "%d %b %Y")
        if gran == "W":
            label = "Week of " + label
        periods.append({
            "label": label, "total": round(float(g["amount"].sum()), 2), "count": int(len(g)),
            "top_shops": [{"merchant": m, "amount": round(float(v), 2)} for m, v in top.items()],
            "unusual_count": int(len(flagged)), "unusual_total": round(float(flagged["amount"].sum()), 2),
            "partial": bool(gran == "M" and per_key == per.max() and last_ts.day < last_ts.days_in_month - 3),
        })
    periods.reverse()   # newest first
    df = df.drop(columns="_unusual")

    # subscriptions: how many times paid, total paid so far, and a confidence tier
    from subscription_model import detect_subscriptions as _detect, cadence_days as _cadence_days
    probe = df.assign(user_id=0)
    detected = _detect(probe)
    conf_by_shop = {r["merchant"]: r["confidence"] for _, r in detected[detected["is_subscription"] == 1].iterrows()}
    sub_rows, monthly_cost = [], 0.0
    for sb in subs:
        rows = df[df["merchant"] == sb.merchant]["amount"]
        n_paid = int(len(rows))
        sub_period = _cadence_days(sb.cadence)
        monthly_cost += sb.avg_amount * 30.44 / sub_period
        first = df[df["merchant"] == sb.merchant]["timestamp"].min()
        sub_rows.append({
            "merchant": sb.merchant, "avg_amount": sb.avg_amount, "cadence": sb.cadence,
            "next_expected_payment": str(sb.next_expected_payment) if sb.next_expected_payment else None,
            "times_paid": n_paid, "total_paid": round(float(rows.sum()), 2),
            "confidence": conf_by_shop.get(sb.merchant, "Low"),
            "since": first.strftime("%b %Y"),
        })

    # unusual payments, made easy to read: how big next to your usual payment,
    # which month, and a short summary the dashboard can lead with
    typical = float(df["amount"].median()) or 1.0
    latest_month = df["timestamp"].max().to_period("M")
    alert_rows = []
    for a in sorted(alerts, key=lambda x: x.timestamp, reverse=True):
        ts = pd.Timestamp(a.timestamp)
        alert_rows.append({
            "transaction_id": a.transaction_id, "amount": a.amount, "merchant": a.merchant,
            "timestamp": str(a.timestamp), "anomaly_score": a.anomaly_score, "explanation": a.explanation,
            "multiple": round(a.amount / typical, 1),
            "month": ts.strftime("%B %Y"), "is_latest_month": ts.to_period("M") == latest_month,
        })
    by_month = {}
    for r in alert_rows:
        m = by_month.setdefault(r["month"], {"month": r["month"], "count": 0, "total": 0.0, "is_latest_month": r["is_latest_month"]})
        m["count"] += 1
        m["total"] += r["amount"]
    biggest = max(alert_rows, key=lambda r: r["amount"]) if alert_rows else None
    anomaly_summary = {
        "count": len(alert_rows),
        "total_amount": round(sum(r["amount"] for r in alert_rows), 2),
        "share_pct": round(100 * sum(r["amount"] for r in alert_rows) / total_spent, 1) if total_spent else 0,
        "typical_payment": round(typical, 2),
        "biggest": biggest,
        "by_month": [dict(v, total=round(v["total"], 2)) for v in by_month.values()],
    }

    return {
        "credits": credit_summary(db, user_id),
        "anomaly_summary": anomaly_summary,
        "profile": {
            "total_transactions": profile.total_transactions if profile else len(txns),
            "average_spending": profile.average_spending if profile else None,
            "median_spending": profile.median_spending if profile else None,
            "has_personal_model": profile.has_personal_model if profile else False,
            "last_updated": str(profile.last_updated) if profile else None,
        },
        "total_spending": round(float(df["amount"].sum()), 2),
        "anomalies": alert_rows,
        "subscriptions": sub_rows,
        "subscription_summary": {
            "monthly_cost": round(monthly_cost, 2),
            "yearly_cost": round(monthly_cost * 12, 2),
            "total_paid_so_far": round(sum(r["total_paid"] for r in sub_rows), 2),
        },
        "spending_by_merchant": shops,
        "spending_trend": trend,
        "periods": periods,
        "spending_by_category": by_category,
        "spending_by_merchant_top10": by_merchant,
        "period_analysis": period,
        "insights": insights,
        "model_status": {
            "has_personal_model": profile.has_personal_model if profile else False,
            "trained_on_count": profile.trained_on_count if profile else 0,
            "model_versions_count": len(model_versions),
            "latest_version": model_versions[0].version if model_versions else None,
        },
        "drift_status": {
            "severity": latest_drift.severity if latest_drift else "none",
            "drift_detected": latest_drift.drift_detected if latest_drift else False,
            "last_checked": str(latest_drift.created_at) if latest_drift else None,
        },
    }


def get_recent_transactions(db: Session, user_id: int, limit: int = 50):
    txns = (
        db.query(models.Transaction)
        .filter(models.Transaction.user_id == user_id)
        .order_by(models.Transaction.timestamp.desc())
        .limit(limit)
        .all()
    )
    from categorizer import fill_categories
    cats = fill_categories([t.merchant for t in txns], [t.category for t in txns])
    return [{
        "transaction_id": t.transaction_id, "timestamp": str(t.timestamp), "amount": t.amount,
        "merchant": t.merchant, "category": c, "payment_mode": t.payment_mode,
    } for t, c in zip(txns, cats)]
