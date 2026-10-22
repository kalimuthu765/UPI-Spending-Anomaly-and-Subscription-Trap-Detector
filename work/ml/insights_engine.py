"""
Insights Engine — Sections 10 & 16 of the master spec.

Two jobs:
  1. detect_period(df) — figures out how much history the user uploaded
     and picks the right analysis granularity. Never assumes 6 months.
  2. generate_insights(df, period, subscriptions) — turns real numbers
     into natural-language sentences ("Amazon is your highest-spending
     merchant"). Every sentence is computed from the data actually
     present — nothing is a hardcoded template with blanks filled in
     from a fixed assumption about the period length.
"""

import pandas as pd
import numpy as np


def _month_name(p):
    return p.start_time.strftime("%B %Y")


def _complete_months(df, by_month):
    """Months fully covered by the data. The first/last month of an upload is
    often partial (e.g. a statement ending on the 14th); comparing a half
    month with full ones gives nonsense like 'you spent 96% less'."""
    first, last = df["timestamp"].min(), df["timestamp"].max()
    keep = []
    for per in by_month.index:
        starts_ok = not (per == first.to_period("M") and first.day > 5)
        ends_ok = not (per == last.to_period("M") and last.day < per.days_in_month - 3)
        if starts_ok and ends_ok:
            keep.append(per)
    return by_month.loc[keep]


def detect_period(df):
    """Returns the granularity tier + the correctly-grained top-spending
    bucket, so a 1-week upload gets 'highest day' while a 1-year upload
    gets 'highest month' — never the wrong comparison for the data."""
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1

    if span_days <= 10:
        tier = "week_or_less"
    elif span_days <= 45:
        tier = "month"
    elif span_days <= 330:
        tier = "several_months"
    else:
        tier = "year_plus"

    result = {"span_days": int(span_days), "tier": tier,
              "start": str(df["timestamp"].min().date()), "end": str(df["timestamp"].max().date())}

    if tier == "week_or_less":
        by_day = df.groupby(df["timestamp"].dt.date)["amount"].sum()
        top = by_day.idxmax()
        result["highest_period_label"] = "day"
        result["highest_period"] = str(top)
        result["highest_period_amount"] = round(float(by_day.max()), 2)
        result["highest_period_weekday"] = pd.Timestamp(top).strftime("%A")

    elif tier == "month":
        df = df.copy()
        df["week_of_month"] = ((df["timestamp"].dt.day - 1) // 7) + 1
        by_week = df.groupby("week_of_month")["amount"].sum()
        top = by_week.idxmax()
        result["highest_period_label"] = "week"
        result["highest_period"] = f"Week {int(top)}"
        result["highest_period_amount"] = round(float(by_week.max()), 2)

    else:  # several_months or year_plus
        by_month = df.groupby(df["timestamp"].dt.to_period("M"))["amount"].sum()
        top = by_month.idxmax()
        result["highest_period_label"] = "month"
        result["highest_period"] = _month_name(top)
        result["highest_period_amount"] = round(float(by_month.max()), 2)
        full = _complete_months(df, by_month)
        if len(full) >= 3:
            result["average_monthly_spending"] = round(float(full.mean()), 2)

        if tier == "year_plus":
            df = df.copy()
            df["quarter"] = df["timestamp"].dt.to_period("Q")
            by_quarter = df.groupby("quarter")["amount"].sum()
            result["highest_quarter"] = str(by_quarter.idxmax())
            result["highest_quarter_amount"] = round(float(by_quarter.max()), 2)

    return result


def generate_insights(df, period, subscriptions_df=None):
    """Natural-language sentences, each computed from the actual data —
    never shown if the underlying comparison isn't supported by the
    available period (e.g. no 'vs last month' on a 1-week upload)."""
    insights = []
    total = df["amount"].sum()

    # highest-spending period (phrased per the detected granularity)
    label = period["highest_period_label"]
    insights.append(
        f"Your highest-spending {label} was {period['highest_period']}"
        f" (₹{period['highest_period_amount']:,.0f})."
    )

    # merchant ranking
    by_merchant = df.groupby("merchant")["amount"].agg(total="sum", count="count", average="mean")
    if len(by_merchant):
        top_m = by_merchant.sort_values("total", ascending=False).iloc[0]
        top_m_name = by_merchant["total"].idxmax()
        n = int(top_m["count"])
        detail = f"{n} payments, about ₹{top_m['average']:,.0f} each" if n > 1 else "a single payment"
        insights.append(f"Most of your money went to {top_m_name}: ₹{top_m['total']:,.0f} over {detail}.")

    # category ranking
    by_category = df.groupby("category")["amount"].sum()
    if len(by_category):
        top_cat = by_category.idxmax()
        pct = by_category.max() / total * 100 if total else 0
        insights.append(f"{top_cat} is your top spending category, at {pct:.0f}% of total spending.")

    # weekend vs weekday (only meaningful with >= 2 weeks of data)
    if period["span_days"] >= 14:
        weekend_avg = df[df["timestamp"].dt.dayofweek >= 5]["amount"].mean()
        weekday_avg = df[df["timestamp"].dt.dayofweek < 5]["amount"].mean()
        if pd.notna(weekend_avg) and pd.notna(weekday_avg) and weekday_avg > 0:
            if weekend_avg > weekday_avg * 1.15:
                insights.append(
                    f"Weekend payments average ₹{weekend_avg:,.0f}, higher than the ₹{weekday_avg:,.0f} "
                    f"you spend on a typical weekday payment."
                )

    # trend vs personal average (only meaningful with several months of data)
    if period["tier"] in ("several_months", "year_plus"):
        by_month = _complete_months(df, df.groupby(df["timestamp"].dt.to_period("M"))["amount"].sum())
        if len(by_month) >= 3:
            latest_month_amt = by_month.iloc[-1]
            avg = float(by_month.iloc[:-1].mean())   # compare against the OTHER complete months
            if avg > 0:
                pct_diff = (latest_month_amt - avg) / avg * 100
                if abs(pct_diff) >= 10:
                    direction = "more" if pct_diff > 0 else "less"
                    insights.append(
                        f"In {_month_name(by_month.index[-1])} you spent {abs(pct_diff):.0f}% {direction} "
                        f"than in your earlier full months (about ₹{avg:,.0f} a month)."
                    )

    # recurring payments summary (non-judgmental, per spec section 15)
    if subscriptions_df is not None and len(subscriptions_df):
        monthly_total = (subscriptions_df["monthly_cost"].sum() if "monthly_cost" in subscriptions_df else subscriptions_df["avg_amount"].sum())
        k = len(subscriptions_df)
        insights.append(
            f"{k} {'payment repeats' if k == 1 else 'payments repeat'} on a schedule, about "
            f"₹{monthly_total:,.0f} a month in total. Worth a look if you no longer use any of them."
        )

    return insights
