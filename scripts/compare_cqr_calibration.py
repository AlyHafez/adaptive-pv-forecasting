"""
Compare the existing static CQR calibration (single q_hat fit once from the
first 30 calendar days) against a rolling alternative (q_hat recomputed
periodically from a trailing calibration window), purely as a post-hoc
diagnostic.

Does NOT retrain anything. Reads the already-computed ensemble CSVs in
results/ and reuses the same nonconformity score / quantile-level formula as
src.models.eval_residual.cqr_calibration, so the two are directly comparable.

Usage:
    python -m scripts.compare_cqr_calibration pretrained 1
    python -m scripts.compare_cqr_calibration pretrained_shaded 1
    python -m scripts.compare_cqr_calibration pretrained_drifted 2
"""
import sys
import logging
import numpy as np
import pandas as pd

from src.config import file_config
from src.evaluation.metrics import compute_probabilistic
from src.models.eval_residual import cqr_calibration, rolling_cqr_calibration

logging.basicConfig(level=logging.INFO)


def evaluate(df: pd.DataFrame, label: str, calibration_days: int | None = None) -> dict:
    """Score calibrated intervals with the existing compute_probabilistic metric.
    If calibration_days is given, also reports test-period-only coverage
    (excluding the warm-up window) to avoid the slight optimistic bias of
    scoring the calibration set on itself."""
    d = df.copy()
    d["corrected_lower"] = d["calibrated_lower"]
    d["corrected_upper"] = d["calibrated_upper"]
    metrics = compute_probabilistic(d, "corrected_median")
    metrics["label"] = label

    if calibration_days is not None:
        dates = sorted(pd.to_datetime(d["date"]).dt.date.unique())
        test_dates = set(dates[calibration_days:])
        d_test = d[pd.to_datetime(d["date"]).dt.date.isin(test_dates)]
        test_metrics = compute_probabilistic(d_test, "corrected_median")
        metrics["coverage_test_only"] = test_metrics["coverage"]

    return metrics


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrained"
    house = sys.argv[2] if len(sys.argv) > 2 else "1"

    suffix = "" if house == "1" else "2"
    fname_map = {
        "pretrained": f"predictions_rolling_pretrained_ensemble{suffix}.csv",
        "pretrained_drifted": f"predictions_rolling_pretrained_drifted_ensemble{suffix}.csv",
        "pretrained_shaded": f"predictions_rolling_pretrained_shaded_ensemble{suffix}.csv",
    }
    if mode not in fname_map:
        raise ValueError(f"Unknown mode: {mode}")

    path = f"{file_config.results_dir}/{fname_map[mode]}"
    logging.info(f"Loading {path}")
    ensemble_results = pd.read_csv(path)

    CALIBRATION_DAYS = 30
    TARGET_COVERAGE = 0.80

    # --- static baseline (current pipeline behaviour) ---
    q_hat_static, static_calibrated = cqr_calibration(
        ensemble_results, target_coverage=TARGET_COVERAGE, calibration_days=CALIBRATION_DAYS
    )
    static_metrics = evaluate(static_calibrated, "static (fixed, Jan-only calibration)", CALIBRATION_DAYS)

    # --- rolling alternative ---
    rolling_calibrated, q_hat_log = rolling_cqr_calibration(
        ensemble_results, target_coverage=TARGET_COVERAGE,
        calibration_days=CALIBRATION_DAYS, recalibrate_every=7,
    )
    rolling_metrics = evaluate(rolling_calibrated, "rolling (30-day trailing, refit weekly)", CALIBRATION_DAYS)

    summary = pd.DataFrame([static_metrics, rolling_metrics])[
        ["label", "coverage", "coverage_test_only", "interval_width",
         "pinball_q10", "pinball_q50", "pinball_q90", "pinball_mean"]
    ]

    print(f"\n=== CQR comparison: mode={mode}, house={house}, target coverage={TARGET_COVERAGE} ===")
    print(summary.to_string(index=False))
    print(f"\nstatic q_hat (fixed, computed once):        {q_hat_static:.4f}")
    print(f"rolling q_hat range (refit every 7 days):    "
          f"{q_hat_log['q_hat'].min():.4f} - {q_hat_log['q_hat'].max():.4f} "
          f"(mean {q_hat_log['q_hat'].mean():.4f}, {len(q_hat_log)} refits)")

    static_gap = abs(static_metrics["coverage_test_only"] - TARGET_COVERAGE)
    rolling_gap = abs(rolling_metrics["coverage_test_only"] - TARGET_COVERAGE)
    print(f"\n|coverage - target| (test period): static={static_gap:.4f}, rolling={rolling_gap:.4f}")
    if rolling_gap < static_gap:
        print("-> rolling calibration is closer to nominal target coverage.")
    else:
        print("-> static calibration is already at (or closer to) nominal target coverage here.")

    out_dir = file_config.results_dir
    summary.to_csv(f"{out_dir}/cqr_comparison_{mode}_h{house}.csv", index=False)
    q_hat_log.to_csv(f"{out_dir}/cqr_rolling_qhat_log_{mode}_h{house}.csv", index=False)
    logging.info(f"Saved comparison to {out_dir}/cqr_comparison_{mode}_h{house}.csv")
