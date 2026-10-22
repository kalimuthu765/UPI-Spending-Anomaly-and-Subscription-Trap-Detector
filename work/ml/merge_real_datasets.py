"""
Merge Real Datasets -> Calibration Stats
------------------------------------------
Per the plan (Section 3): don't blindly download-and-merge the three
Kaggle datasets row-by-row (they have different schemas/grains and
overlap only loosely). Instead, extract real DISTRIBUTIONS from each
and use them to calibrate the synthetic generator so
final_transactions.csv reflects real-world UPI behavior:

  Dataset 1 (upi_transactions_2024.csv, 250k rows)
      -> category shares, per-category amount mean/std, real fraud rate,
         hour-of-day / day-of-week shape  [PRIMARY calibration source]
  Dataset 2 (upi_transaction_insights_dataset.csv, 500 rows)
      -> payment_mode split (QR/Contact/Online/AutoPay), fills amount
         stats for categories dataset 1 lacks (e.g. Online Services)
  Dataset 3 (Digital_Payments_2025_Dataset.csv, monthly aggregates)
      -> month-over-month seasonal growth curve (real total tx value
         per period) to weight WHEN synthetic transactions occur

Output: data/processed/calibration_stats.json, consumed by
generate_synthetic_data.py to replace its arbitrary uniform ranges.
"""

import json
import pandas as pd
import numpy as np

RAW = "data/raw"
OUT = "data/processed/calibration_stats.json"

# our canonical category set <- real dataset category names
CATEGORY_MAP_DS1 = {
    "Food": "Food & Dining", "Grocery": "Groceries", "Utilities": "Bills & Utilities",
    "Transport": "Travel", "Education": "Education", "Entertainment": "Entertainment",
    "Fuel": "Fuel", "Healthcare": "Healthcare", "Shopping": "Shopping", "Other": "Other",
}
CATEGORY_MAP_DS2 = {
    "Groceries": "Groceries", "Food & Dining": "Food & Dining", "Health & Pharmacy": "Healthcare",
    "Travel": "Travel", "Bills & Utilities": "Bills & Utilities", "Online Services": "Subscriptions",
    "Other": "Other", "Fuel": "Fuel", "Shopping": "Shopping", "Entertainment": "Entertainment",
}


def calibrate_from_ds1():
    df = pd.read_csv(f"{RAW}/upi_transactions_2024.csv")
    df["category"] = df["merchant_category"].map(CATEGORY_MAP_DS1)

    shares = (df["category"].value_counts(normalize=True)).to_dict()

    amount_stats = {}
    for cat, g in df.groupby("category"):
        amt = g["amount (INR)"]
        # fit a lognormal in amount space (amounts are right-skewed, always positive)
        log_amt = np.log(amt.clip(lower=1))
        amount_stats[cat] = {
            "log_mean": float(log_amt.mean()),
            "log_std": float(log_amt.std()),
            "real_mean": float(amt.mean()),
            "real_median": float(amt.median()),
        }

    fraud_rate = float(df["fraud_flag"].mean())

    hour_weights = (df["hour_of_day"].value_counts(normalize=True).sort_index()).to_dict()
    dow_weights = (df["day_of_week"].value_counts(normalize=True)).to_dict()

    return {
        "category_shares": shares,
        "category_amount_stats": amount_stats,
        "real_fraud_rate": fraud_rate,
        "hour_weights": hour_weights,
        "day_of_week_weights": dow_weights,
        "source_rows": len(df),
    }


def calibrate_from_ds2():
    df = pd.read_csv(f"{RAW}/upi_transaction_insights_dataset.csv")
    payment_mode_weights = (df["payment_mode"].value_counts(normalize=True)).to_dict()

    df["category"] = df["merchant_category"].map(CATEGORY_MAP_DS2)
    extra_amount_stats = {}
    for cat, g in df.groupby("category"):
        if pd.isna(cat):
            continue
        amt = g["amount"]
        log_amt = np.log(amt.clip(lower=1))
        extra_amount_stats[cat] = {
            "log_mean": float(log_amt.mean()),
            "log_std": float(log_amt.std()),
            "real_mean": float(amt.mean()),
            "real_median": float(amt.median()),
        }

    return {
        "payment_mode_weights": payment_mode_weights,
        "extra_category_amount_stats": extra_amount_stats,  # fills gaps (e.g. Subscriptions)
        "source_rows": len(df),
    }


def calibrate_from_ds3():
    df = pd.read_csv(f"{RAW}/Digital_Payments_2025_Dataset.csv")
    monthly = df.groupby("Period")["TotalTxValue (Cr)"].sum().sort_index()
    weights = (monthly / monthly.sum()).to_dict()
    return {
        "monthly_seasonality_weights": {int(k): float(v) for k, v in weights.items()},
        "source_rows": len(df),
    }


def main():
    print("Calibrating from upi_transactions_2024.csv (main, 250k rows)...")
    ds1 = calibrate_from_ds1()
    print(f"  category shares: { {k: round(v,3) for k,v in ds1['category_shares'].items()} }")
    print(f"  real fraud rate: {ds1['real_fraud_rate']*100:.3f}%")

    print("Calibrating from upi_transaction_insights_dataset.csv (payment modes)...")
    ds2 = calibrate_from_ds2()
    print(f"  payment mode weights: {ds2['payment_mode_weights']}")

    print("Calibrating from Digital_Payments_2025_Dataset.csv (seasonality)...")
    ds3 = calibrate_from_ds3()
    print(f"  monthly weights: { {k: round(v,3) for k,v in ds3['monthly_seasonality_weights'].items()} }")

    # merge: ds1's category amount stats are primary; ds2 fills categories ds1 lacks
    merged_amount_stats = dict(ds1["category_amount_stats"])
    for cat, stats in ds2["extra_category_amount_stats"].items():
        if cat not in merged_amount_stats:
            merged_amount_stats[cat] = stats

    calibration = {
        "category_shares": ds1["category_shares"],
        "category_amount_stats": merged_amount_stats,
        "real_fraud_rate": ds1["real_fraud_rate"],
        "hour_weights": ds1["hour_weights"],
        "day_of_week_weights": ds1["day_of_week_weights"],
        "payment_mode_weights": ds2["payment_mode_weights"],
        "monthly_seasonality_weights": ds3["monthly_seasonality_weights"],
        "sources": {
            "upi_transactions_2024.csv": ds1["source_rows"],
            "upi_transaction_insights_dataset.csv": ds2["source_rows"],
            "Digital_Payments_2025_Dataset.csv": ds3["source_rows"],
        },
    }

    with open(OUT, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nSaved calibration stats -> {OUT}")


if __name__ == "__main__":
    main()
