"""
Phase 11 — Drift Detection
Core math, reusable at both the global (whole-dataset) level and the
per-user level. Compares a BASELINE distribution (what the model was
trained on) against a CURRENT distribution (recent transactions) and
decides whether the shift is big enough to warrant retraining.

Two tests, per the spec:
  - PSI (Population Stability Index) — the standard drift metric.
      < 0.1  no significant drift
      0.1-0.25  moderate drift, worth watching
      > 0.25  significant drift, retrain
  - KS test (Kolmogorov-Smirnov) — statistical test for "are these two
    samples from the same distribution?" p < 0.05 -> reject "same dist."
"""

import numpy as np
from scipy.stats import ks_2samp

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05
MIN_SAMPLES_FOR_DRIFT_CHECK = 12   # PSI/KS are unstable below this; too few points
                                     # for a small "current" batch produces false positives


def compute_psi(baseline, current, n_bins=10):
    """Population Stability Index between two 1-D numeric arrays."""
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(baseline) < MIN_SAMPLES_FOR_DRIFT_CHECK or len(current) < MIN_SAMPLES_FOR_DRIFT_CHECK:
        return None

    # With few samples, 10 bins leaves ~1-2 points per bin and PSI is mostly
    # noise (even two identical distributions can score > 0.25). Use fewer
    # bins until there is enough data to support more.
    n_bins = int(max(3, min(n_bins, min(len(baseline), len(current)) // 5)))
    breakpoints = np.quantile(baseline, np.linspace(0, 1, n_bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return None  # not enough spread in baseline to bin meaningfully

    base_counts, _ = np.histogram(baseline, bins=breakpoints)
    curr_counts, _ = np.histogram(current, bins=breakpoints)

    base_pct = np.clip(base_counts / max(len(baseline), 1), 1e-4, None)
    curr_pct = np.clip(curr_counts / max(len(current), 1), 1e-4, None)

    psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    return round(psi, 4)


def compute_ks_test(baseline, current):
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(baseline) < MIN_SAMPLES_FOR_DRIFT_CHECK or len(current) < MIN_SAMPLES_FOR_DRIFT_CHECK:
        return None, None
    stat, pvalue = ks_2samp(baseline, current)
    return round(float(stat), 4), round(float(pvalue), 4)


def assess_drift(baseline_amounts, current_amounts):
    """
    Returns a dict describing whether retraining is warranted, combining
    PSI (magnitude of the shift) and the KS test (statistical confidence
    the distributions differ) — matches the spec's Phase 11 diagram.
    """
    psi = compute_psi(baseline_amounts, current_amounts)
    ks_stat, ks_pvalue = compute_ks_test(baseline_amounts, current_amounts)

    baseline_mean = float(np.mean(baseline_amounts)) if len(baseline_amounts) else None
    current_mean = float(np.mean(current_amounts)) if len(current_amounts) else None

    if psi is None:
        severity = "insufficient_data"
        drift_detected = False
    elif psi >= PSI_SIGNIFICANT_THRESHOLD and (ks_pvalue is None or ks_pvalue < KS_PVALUE_THRESHOLD):
        # a big PSI alone isn't enough on small samples — the KS test must
        # also agree the two samples really differ
        severity = "significant"
        drift_detected = True
    elif psi >= PSI_SIGNIFICANT_THRESHOLD:
        severity = "moderate"
        drift_detected = False
    elif psi >= PSI_MODERATE_THRESHOLD:
        severity = "moderate"
        drift_detected = (ks_pvalue is not None and ks_pvalue < KS_PVALUE_THRESHOLD)
    else:
        severity = "none"
        drift_detected = False

    return {
        "psi": psi,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "baseline_mean": round(baseline_mean, 2) if baseline_mean is not None else None,
        "current_mean": round(current_mean, 2) if current_mean is not None else None,
        "severity": severity,
        "drift_detected": drift_detected,
    }


def assess_sustained_shift(baseline_amounts, new_amounts):
    """Tells a one-off spike apart from a real lifestyle change (e.g. college
    -> first job). A single heavy-spending month moves the whole new batch,
    but a real shift shows up in BOTH the earlier and the later half of the
    new transactions, in the same direction. We need enough new data to
    check both halves (2 x MIN_SAMPLES_FOR_DRIFT_CHECK); with less than that
    we can't tell yet, so the answer is 'not proven sustained' and the
    caller keeps the old model and just flags the spike as anomalies."""
    new_amounts = np.asarray(new_amounts, dtype=float)
    baseline_amounts = np.asarray(baseline_amounts, dtype=float)
    if len(new_amounts) < 2 * MIN_SAMPLES_FOR_DRIFT_CHECK:
        return False
    half = len(new_amounts) // 2
    early, late = new_amounts[:half], new_amounts[half:]
    if not (assess_drift(baseline_amounts, early)["drift_detected"]
            and assess_drift(baseline_amounts, late)["drift_detected"]):
        return False
    # practical significance too: both halves must sit at least 1.5x above (or
    # below) the user's usual level, in the same direction
    base_med = max(np.median(baseline_amounts), 1.0)
    r_early, r_late = np.median(early) / base_med, np.median(late) / base_med
    return (r_early >= 1.5 and r_late >= 1.5) or (r_early <= 1 / 1.5 and r_late <= 1 / 1.5)


def latest_window_drifted(baseline_amounts, new_amounts):
    """True if the most recent transactions (last MIN_SAMPLES_FOR_DRIFT_CHECK
    or more) are still drifted from the baseline — used so routine refreshes
    never absorb a spike that is still in progress."""
    new_amounts = np.asarray(new_amounts, dtype=float)
    if len(new_amounts) < MIN_SAMPLES_FOR_DRIFT_CHECK:
        return False
    return assess_drift(baseline_amounts, new_amounts[-MIN_SAMPLES_FOR_DRIFT_CHECK:])["drift_detected"]
