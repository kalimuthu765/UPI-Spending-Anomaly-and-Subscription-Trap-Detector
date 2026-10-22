"""
Phase 5 — Subscription Detection
Separate module from the anomaly detector. Flags (user, merchant) pairs
as recurring subscriptions when merchant repeats, amount is consistent,
and the interval is close to a monthly (or weekly/annual) cadence.
Outputs a dashboard-ready table with the next expected payment date.
"""

import pandas as pd
import numpy as np
from categorizer import SUBSCRIPTION_HINTS, HABIT_CATEGORIES, guess_category

MIN_AMOUNT_CONSISTENCY = 0.85  # 1 - coefficient_of_variation
MIN_PERIOD_DAYS = 6            # daily/near-daily same-amount buys (a tea shop) are habits, not subscriptions

# names for the usual billing periods; anything else becomes "every N weeks/days"
_NAMED = [(7, "weekly"), (14, "fortnightly"), (30, "monthly"), (61, "every 2 months"),
          (91, "quarterly"), (182, "every 6 months"), (365, "yearly")]


def cadence_label(days):
    """Turn a billing period in days into plain words. Any period is allowed:
    a week, 10 weeks, 3 months, a year ..."""
    days = float(days)
    for d, name in _NAMED:
        if abs(days - d) <= max(2.0, 0.08 * d):
            return name
    if days < 100:
        w = days / 7
        if abs(w - round(w)) <= 0.2:
            return f"every {int(round(w))} weeks"
        if days < 60:
            return f"every {int(round(days))} days"
    if days > 500:
        yrs = days / 365
        if abs(yrs - round(yrs)) <= 0.12:
            return f"every {int(round(yrs))} years"
    mo = days / 30.44
    if mo < 24 and abs(mo - round(mo)) <= 0.12 * max(1, round(mo)) ** 0.5:
        return f"every {int(round(mo))} months"
    return f"every {int(round(days))} days"


def cadence_days(label):
    """Inverse of cadence_label (used for next-payment date and monthly cost)."""
    for d, name in _NAMED:
        if label == name:
            return d
    import re
    m = re.match(r"every (\d+) (days|weeks|months|years)", str(label or ""))
    if m:
        return round(int(m.group(1)) * {"days": 1, "weeks": 7, "months": 30.44, "years": 365}[m.group(2)])
    return 30


def classify_cadence(avg_interval):
    return cadence_label(avg_interval)


def detect_subscriptions(df):
    """A (user, shop) pair is a subscription when charges repeat at a steady
    interval (whatever that interval is) for a steady amount. Works from the
    raw timestamps, so it does not matter whether the person pays weekly,
    monthly, every 10 weeks or once a year."""
    rows = []
    for (uid, merchant), g in df.groupby(["user_id", "merchant"]):
        g = g.sort_values("timestamp")
        days = g["timestamp"].dt.normalize().drop_duplicates()
        n = len(days)
        amounts = g["amount"].astype(float)
        mean_amt = float(amounts.mean())
        consistency = float(max(0.0, 1 - amounts.std(ddof=0) / mean_amt)) if mean_amt > 0 else 0.0
        named_plan = bool(SUBSCRIPTION_HINTS.search(str(merchant)))
        habit = guess_category(merchant) in HABIT_CATEGORIES and not named_plan   # petrol, groceries, meals ...
        row = {"user_id": uid, "merchant": merchant, "merchant_repeat_count": int(len(g)),
               "amount_consistency": consistency, "avg_amount": mean_amt,
               "last_txn": g["timestamp"].max(), "category": g["category"].iloc[0] if "category" in g else None,
               "recurring_interval": np.nan, "cadence": None, "is_subscription": 0, "confidence": None,
               "period_days": np.nan, "interval_regularity": 0.0}
        if n >= 2:
            gaps = np.diff(days.values).astype("timedelta64[D]").astype(float)
            period = float(np.median(gaps))
            row["recurring_interval"] = period
            if period >= MIN_PERIOD_DAYS:
                # how far the real gaps stray from the typical gap (calendar months vary 28-31 days)
                dev = float(np.max(np.abs(gaps - period)) / period)
                regularity = max(0.0, 1 - dev)
                row["interval_regularity"] = regularity
                tol = 0.20 if period < 80 else 0.12
                long_cycle = period >= 80
                enough = n >= 3 or (long_cycle and n >= 2)
                steady_amount = consistency >= (0.98 if n == 2 else MIN_AMOUNT_CONSISTENCY)
                if named_plan:
                    # a shop that is clearly a plan (Netflix, Jio recharge, broadband ...):
                    # accept a looser rhythm and two charges, but never claim High confidence
                    tol = max(tol, 0.40)
                    enough = n >= 2
                    steady_amount = consistency >= 0.80
                if enough and dev <= tol and steady_amount and not habit:
                    row.update(is_subscription=1, period_days=period, cadence=cadence_label(period))
                    if named_plan and (n < 4 or dev > 0.20):
                        row["confidence"] = "Low" if n < 3 else "Medium"
                    elif n >= 5 and consistency >= 0.95 and dev <= 0.12:
                        row["confidence"] = "High"
                    elif n >= 3 and consistency >= 0.90 and dev <= 0.20 and (n >= 4 or long_cycle):
                        row["confidence"] = "Medium"
                    else:
                        row["confidence"] = "Low"
        rows.append(row)

    grouped = pd.DataFrame(rows)
    if grouped.empty:
        grouped = pd.DataFrame(columns=["user_id", "merchant", "avg_amount", "last_txn", "cadence",
                                        "is_subscription", "confidence", "next_expected_payment",
                                        "recurring_probability", "monthly_cost", "period_days"])
        return grouped
    grouped["recurring_probability"] = np.where(
        grouped["is_subscription"] == 1,
        np.clip(grouped["amount_consistency"] * grouped["interval_regularity"].clip(lower=0.5), 0, 1),
        np.clip(grouped["amount_consistency"] * 0.3, 0, 1))
    grouped["next_expected_payment"] = grouped.apply(
        lambda r: pd.Timestamp(r["last_txn"]) + pd.Timedelta(days=float(r["period_days"]))
        if r["is_subscription"] else pd.NaT, axis=1)
    grouped["monthly_cost"] = np.where(
        grouped["is_subscription"] == 1, grouped["avg_amount"] * 30.44 / grouped["period_days"].fillna(30), 0.0)
    return grouped.sort_values(["user_id", "recurring_probability"], ascending=[True, False])


def evaluate_against_declared(subs_df):
    """Sanity check against the recurring_flag the generator assigned —
    real deployments won't have this ground truth, but it's useful here."""
    detected = subs_df["is_subscription"]
    declared = (subs_df["declared_recurring"] > 0)
    tp = ((detected == 1) & declared).sum()
    fp = ((detected == 1) & ~declared).sum()
    fn = ((detected == 0) & declared).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def run(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    subs = detect_subscriptions(df)

    metrics = evaluate_against_declared(subs)
    print("Subscription detector evaluation vs. generator's ground-truth recurring_flag:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    detected_only = subs[subs["is_subscription"] == 1]
    detected_only.to_csv(output_path, index=False)
    print(f"\n{len(detected_only)} subscriptions detected across "
          f"{detected_only['user_id'].nunique()} users -> {output_path}")

    print("\nSample detections:")
    for _, r in detected_only.head(6).iterrows():
        print(f"  [{r['user_id']}] {r['merchant']} (Rs.{r['avg_amount']:.0f}, {r['cadence']}) "
              f"-> next expected {r['next_expected_payment'].date()}")

    return subs, metrics


if __name__ == "__main__":
    run(
        input_path="data/processed/transactions_featured.csv",
        output_path="data/processed/subscriptions_detected.csv",
    )
