"""
Synthetic UPI Transaction Dataset Generator
--------------------------------------------
Produces data/synthetic/final_transactions.csv matching the project spec:
  - 1,000 users
  - ~100,000 transactions
  - 12 months of history
  - 10-15 categories, 100+ merchants
  - recurring/subscription behavior baked in
  - injected anomalies for the anomaly-detection model to learn from

CALIBRATED against real data: if data/processed/calibration_stats.json
exists (built by merge_real_datasets.py from the three uploaded Kaggle
datasets), category mix, per-category amount distributions, payment
mode split, and month-by-month seasonality are all drawn from real UPI
behavior instead of arbitrary uniform ranges. Per-user tiers still
scale amounts so personalization (Phase 6) has something to learn.
Falls back to the original arbitrary ranges if calibration is absent.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

RNG = np.random.default_rng(42)

CALIBRATION_PATH = "data/processed/calibration_stats.json"
CALIBRATION = None
if os.path.exists(CALIBRATION_PATH):
    with open(CALIBRATION_PATH) as f:
        CALIBRATION = json.load(f)

N_USERS = 1000
N_MONTHS = 12
START_DATE = datetime(2025, 9, 1)
END_DATE = START_DATE + timedelta(days=30 * N_MONTHS)

CATEGORIES = [
    "Food & Dining", "Groceries", "Shopping", "Travel", "Entertainment",
    "Bills & Utilities", "Healthcare", "Education", "Subscriptions",
    "Fuel", "Rent", "Investments", "Personal Care", "Recharge"
]

# 100+ merchants across categories
MERCHANT_POOL = {
    "Food & Dining": ["Zomato", "Swiggy", "Dominos", "McDonalds", "KFC", "CafeCoffeeDay", "Starbucks", "LocalDhaba"],
    "Groceries": ["BigBasket", "Blinkit", "Zepto", "DMart", "Reliance Fresh", "More Retail"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "Meesho", "Nykaa", "TataCliq"],
    "Travel": ["IRCTC", "MakeMyTrip", "Uber", "Ola", "RedBus", "Goibibo", "IndiGo"],
    "Entertainment": ["BookMyShow", "PVR", "INOX", "Steam"],
    "Bills & Utilities": ["TNEB", "BSNL", "Airtel", "ACT Fibernet", "Jio Fiber", "Municipal Water"],
    "Healthcare": ["Apollo Pharmacy", "PharmEasy", "1mg", "Practo", "Medplus"],
    "Education": ["Udemy", "Coursera", "BYJUS", "Unacademy", "College Fees Portal"],
    "Subscriptions": ["Netflix", "Amazon Prime", "Spotify", "Hotstar", "YouTube Premium", "iCloud", "Google One"],
    "Fuel": ["IndianOil", "BPCL", "HPCL", "Shell"],
    "Rent": ["NoBroker Rent", "Housing.com Rent", "Landlord UPI"],
    "Investments": ["Zerodha", "Groww", "Paytm Money", "PPF Contribution"],
    "Personal Care": ["Nykaa", "Purplle", "Local Salon", "Cult.fit"],
    "Recharge": ["Jio Recharge", "Airtel Recharge", "Vi Recharge", "DTH Recharge"],
}
# pad merchant pool to guarantee 100+ unique merchants
ALL_MERCHANTS = sorted({m for ms in MERCHANT_POOL.values() for m in ms})
extra_needed = max(0, 105 - len(ALL_MERCHANTS))
ALL_MERCHANTS += [f"LocalMerchant_{i}" for i in range(extra_needed)]

PAYMENT_MODES = ["UPI-Collect", "UPI-Intent", "AutoPay"]
TRANSACTION_TYPES = ["Debit", "Credit"]
LOCATIONS = ["Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Delhi", "Pune", "Coimbatore", "Kochi"]

SUBSCRIPTION_CATALOG = [
    ("Netflix", 199), ("Amazon Prime", 299), ("Spotify", 119),
    ("Hotstar", 149), ("YouTube Premium", 139), ("iCloud", 75), ("Google One", 130),
]


def random_dates_in_range(n, start, end):
    """Uniform date sampling (fallback when no seasonality calibration)."""
    delta_days = (end - start).days
    offsets = RNG.integers(0, delta_days, size=n)
    seconds = RNG.integers(0, 86400, size=n)
    return [start + timedelta(days=int(d), seconds=int(s)) for d, s in zip(offsets, seconds)]


def seasonal_dates_in_range(n, start, end, monthly_weights):
    """Sample dates so each of the 12 months gets a share of transactions
    proportional to the real Digital_Payments_2025 monthly volume trend."""
    months = list(range(1, 13))
    weights = np.array([monthly_weights.get(str(m), monthly_weights.get(m, 1 / 12)) for m in months])
    weights = weights / weights.sum()
    month_offsets = RNG.choice(range(12), size=n, p=weights)
    dates = []
    for off in month_offsets:
        day_in_month = int(RNG.integers(0, 28))
        seconds = int(RNG.integers(0, 86400))
        dates.append(start + timedelta(days=int(off) * 30 + day_in_month, seconds=seconds))
    return dates


def gen_user_profiles(n_users):
    tiers = RNG.choice(["low", "mid", "high"], size=n_users, p=[0.5, 0.35, 0.15])
    # tier_multiplier scales the real/base amount distribution up or down
    # per user, so the personalization phase (Phase 6) has signal to learn.
    tier_multiplier = {"low": (0.3, 0.7), "mid": (0.8, 1.3), "high": (1.5, 4.0)}
    base_ranges = {"low": (80, 600), "mid": (300, 2500), "high": (1500, 12000)}
    profiles = []
    for i in range(n_users):
        lo, hi = base_ranges[tiers[i]]
        mlo, mhi = tier_multiplier[tiers[i]]
        profiles.append({
            "user_id": f"U{1000 + i}",
            "tier": tiers[i],
            "typical_low": lo,
            "typical_high": hi,
            "spend_multiplier": round(float(RNG.uniform(mlo, mhi)), 3),
            "home_location": RNG.choice(LOCATIONS),
        })
    return pd.DataFrame(profiles)


def _sample_category(rng_val, category_shares):
    """Weighted category pick using real shares (falls back to remaining
    canonical categories, evenly weighted, for ones real data didn't cover)."""
    covered = list(category_shares.keys())
    covered_weight = sum(category_shares.values())
    uncovered = [c for c in CATEGORIES if c not in covered]
    if uncovered and covered_weight < 1.0:
        remaining = 1.0 - covered_weight
        per_uncovered = remaining / len(uncovered)
        all_cats = covered + uncovered
        all_weights = [category_shares[c] for c in covered] + [per_uncovered] * len(uncovered)
    else:
        all_cats = covered
        all_weights = [category_shares[c] for c in covered]
    all_weights = np.array(all_weights) / sum(all_weights)
    return RNG.choice(all_cats, p=all_weights)


def gen_regular_transactions(users_df, n_transactions):
    rows = []
    picks = RNG.integers(0, len(users_df), size=n_transactions)

    if CALIBRATION:
        dates = seasonal_dates_in_range(
            n_transactions, START_DATE, END_DATE, CALIBRATION["monthly_seasonality_weights"]
        )
        cat_shares = CALIBRATION["category_shares"]
        amount_stats = CALIBRATION["category_amount_stats"]
        pm_weights = CALIBRATION["payment_mode_weights"]
        pm_modes, pm_probs = list(pm_weights.keys()), np.array(list(pm_weights.values()))
        pm_probs = pm_probs / pm_probs.sum()
    else:
        dates = random_dates_in_range(n_transactions, START_DATE, END_DATE)

    for idx in range(n_transactions):
        u = users_df.iloc[picks[idx]]

        if CALIBRATION:
            category = _sample_category(idx, cat_shares)
            stats = amount_stats.get(category)
            if stats:
                # real lognormal shape, scaled by this user's spend tier
                raw = float(RNG.lognormal(stats["log_mean"], max(stats["log_std"], 0.1)))
                amount = round(raw * u["spend_multiplier"], 2)
            else:
                amount = round(float(RNG.uniform(u["typical_low"], u["typical_high"])), 2)
            payment_mode = RNG.choice(pm_modes, p=pm_probs)
        else:
            category = RNG.choice(CATEGORIES)
            amount = round(float(RNG.uniform(u["typical_low"], u["typical_high"])), 2)
            payment_mode = RNG.choice(PAYMENT_MODES, p=[0.55, 0.35, 0.10])

        merchant = RNG.choice(MERCHANT_POOL.get(category, ALL_MERCHANTS))
        rows.append({
            "transaction_id": str(uuid.uuid4()),
            "user_id": u["user_id"],
            "timestamp": dates[idx],
            "amount": max(amount, 1.0),
            "merchant": merchant,
            "category": category,
            "transaction_type": "Debit",
            "payment_mode": payment_mode,
            "location": RNG.choice(LOCATIONS) if RNG.random() < 0.1 else u["home_location"],
            "recurring_flag": 0,
            "subscription_id": "",
            "anomaly_label": 0,
        })
    return rows


def gen_subscription_transactions(users_df):
    """Every user gets 0-4 recurring subscriptions, one charge per month."""
    rows = []
    for _, u in users_df.iterrows():
        n_subs = RNG.choice([0, 1, 2, 3, 4], p=[0.15, 0.30, 0.30, 0.15, 0.10])
        chosen = RNG.choice(len(SUBSCRIPTION_CATALOG), size=n_subs, replace=False) if n_subs else []
        for ci in chosen:
            merchant, base_amount = SUBSCRIPTION_CATALOG[ci]
            sub_id = f"SUB-{u['user_id']}-{merchant[:4].upper()}"
            # anchor day-of-month, charge every ~30 days with small jitter
            anchor_day = int(RNG.integers(1, 28))
            current = START_DATE.replace(day=anchor_day)
            while current < END_DATE:
                jitter_amt = round(base_amount * float(RNG.uniform(0.98, 1.03)), 2)
                jitter_days = int(RNG.integers(-1, 2))
                ts = current + timedelta(days=jitter_days, hours=int(RNG.integers(0, 23)))
                if START_DATE <= ts <= END_DATE:
                    rows.append({
                        "transaction_id": str(uuid.uuid4()),
                        "user_id": u["user_id"],
                        "timestamp": ts,
                        "amount": jitter_amt,
                        "merchant": merchant,
                        "category": "Subscriptions",
                        "transaction_type": "Debit",
                        "payment_mode": "AutoPay",
                        "location": u["home_location"],
                        "recurring_flag": 1,
                        "subscription_id": sub_id,
                        "anomaly_label": 0,
                    })
                current += timedelta(days=30)
    return rows


def inject_anomalies(df, fraction=0.015):
    """Flip a random subset of transactions into anomalies: spike amount,
    unfamiliar merchant, or odd hour — and label them for supervised eval."""
    n = len(df)
    n_anom = int(n * fraction)
    idxs = RNG.choice(n, size=n_anom, replace=False)
    df = df.copy()
    for i in idxs:
        kind = RNG.choice(["amount_spike", "odd_merchant", "odd_hour"])
        if kind == "amount_spike":
            df.at[i, "amount"] = round(df.at[i, "amount"] * float(RNG.uniform(8, 25)), 2)
        elif kind == "odd_merchant":
            df.at[i, "merchant"] = RNG.choice(ALL_MERCHANTS)
            df.at[i, "amount"] = round(df.at[i, "amount"] * float(RNG.uniform(3, 10)), 2)
        else:
            ts = pd.Timestamp(df.at[i, "timestamp"]).replace(hour=int(RNG.integers(1, 5)))
            df.at[i, "timestamp"] = ts
        df.at[i, "anomaly_label"] = 1
    return df


def main():
    print(f"Generating profiles for {N_USERS} users...")
    users_df = gen_user_profiles(N_USERS)

    if CALIBRATION:
        print(f"Calibration loaded: real category shares, amount distributions, "
              f"payment modes, and monthly seasonality from "
              f"{list(CALIBRATION['sources'].keys())}")
        print(f"  (real fraud rate observed: {CALIBRATION['real_fraud_rate']*100:.3f}% — "
              f"anomaly injection below uses a higher rate so the model has enough "
              f"positive examples to learn from)")
    else:
        print("No calibration_stats.json found — using arbitrary uniform ranges. "
              "Run ml/merge_real_datasets.py first to calibrate against real data.")

    print("Generating regular transactions...")
    regular_rows = gen_regular_transactions(users_df, n_transactions=94000)

    print("Generating subscription/recurring transactions...")
    sub_rows = gen_subscription_transactions(users_df)

    all_rows = regular_rows + sub_rows
    df = pd.DataFrame(all_rows)

    # top up / trim to land close to 100k total
    target = 100000
    if len(df) > target:
        df = df.sample(n=target, random_state=42).reset_index(drop=True)
    elif len(df) < target:
        extra = gen_regular_transactions(users_df, n_transactions=target - len(df))
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    print("Injecting anomalies...")
    df = inject_anomalies(df, fraction=0.015)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    out_path = "data/synthetic/final_transactions.csv"
    df.to_csv(out_path, index=False)

    print(f"\nDone. {len(df):,} transactions | {df['user_id'].nunique()} users | "
          f"{df['merchant'].nunique()} merchants | {df['category'].nunique()} categories")
    print(f"Anomalies: {df['anomaly_label'].sum():,} ({df['anomaly_label'].mean()*100:.2f}%)")
    print(f"Recurring: {df['recurring_flag'].sum():,} ({df['recurring_flag'].mean()*100:.2f}%)")
    print(f"Saved to: {out_path}")

    users_df.to_csv("data/synthetic/user_profiles.csv", index=False)


if __name__ == "__main__":
    main()
