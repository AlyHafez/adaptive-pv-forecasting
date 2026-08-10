"""
Pull together the headline forecasting + economic metrics for a supervisor
presentation: NRMSE/MAE table across models and conditions, probabilistic
(pinball/coverage) metrics, the control/economic summary, and a small set
of presentation-ready charts.

Reads only already-computed result CSVs -- no retraining, no GPU needed.

Usage:
    python -m scripts.generate_presentation_metrics
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.config import file_config
from src.evaluation.metrics import compute_metrics, compute_probabilistic

logging.basicConfig(level=logging.INFO)

MODELS = {
    "Naive":        "naive",
    "SARIMA":       "arima_pred",
    "TFT":          "tft_pred",
    "TFT+Residual": "corrected_median",
}

CONDITIONS = {
    "Clean":   ("predictions_rolling_pretrained_ensemble{suffix}.csv",),
    "Drifted": ("predictions_rolling_pretrained_drifted_ensemble{suffix}.csv",),
    "Shaded":  ("predictions_rolling_pretrained_shaded_ensemble{suffix}.csv",),
}

CONTROL_FILES = {
    ("Clean", "1"):   "control_summary_pretrained_h1.csv",
    ("Clean", "2"):   "control_summary_pretrained_h2.csv",
    ("Drifted", "1"): "control_summary_pretrained_drifted_h1.csv",
    ("Drifted", "2"): "control_summary_pretrained_drifted_h2.csv",
    ("Shaded", "1"):  "control_summary_pretrained_shaded_h1.csv",
    ("Shaded", "2"):  "control_summary_pretrained_shaded_h2.csv",
}

COLORS = {
    "Naive":        "#999999",
    "SARIMA":       "#E24A33",
    "TFT":          "#348ABD",
    "TFT+Residual": "#1B7837",
}


def load_ensemble(condition: str, house: str) -> pd.DataFrame:
    suffix = "" if house == "1" else "2"
    fname = CONDITIONS[condition][0].format(suffix=suffix)
    return pd.read_csv(f"{file_config.results_dir}/{fname}")


def build_forecast_table() -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for house in ["1", "2"]:
            df = load_ensemble(condition, house)
            for model_name, col in MODELS.items():
                m = compute_metrics(df, col)
                rows.append({
                    "condition": condition, "house": house, "model": model_name,
                    "RMSE": m["RMSE"], "NRMSE": m["NRMSE"], "MAE": m["MAE"], "MBE": m["MBE"],
                })
    return pd.DataFrame(rows)


def build_probabilistic_table() -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for house in ["1", "2"]:
            df = load_ensemble(condition, house)
            for model_name, target in [("TFT", "tft_pred"), ("TFT+Residual", "corrected_median")]:
                p = compute_probabilistic(df, target)
                rows.append({
                    "condition": condition, "house": house, "model": model_name,
                    "coverage": p["coverage"], "interval_width": p["interval_width"],
                    "pinball_q10": p["pinball_q10"], "pinball_q50": p["pinball_q50"],
                    "pinball_q90": p["pinball_q90"], "pinball_mean": p["pinball_mean"],
                })
    return pd.DataFrame(rows)


def build_control_table() -> pd.DataFrame:
    rows = []
    for (condition, house), fname in CONTROL_FILES.items():
        df = pd.read_csv(f"{file_config.results_dir}/{fname}")
        for _, r in df.iterrows():
            rows.append({
                "condition": condition, "house": house, "scenario": r["scenario"],
                "revenue": r["apparent_rev"], "curtailment_kwh": r.get("curtailment_kwh"),
                "curtailment_rate": r.get("curtailment_rate"), "self_consumption": r.get("self_consumption"),
                "violations": r.get("violations"),
            })
    return pd.DataFrame(rows)


def plot_nrmse_bars(forecast_table: pd.DataFrame, house: str, outfile: str):
    conditions = list(CONDITIONS.keys())
    models = list(MODELS.keys())
    x = np.arange(len(conditions))
    width = 0.19

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, model_name in enumerate(models):
        vals = [
            forecast_table[
                (forecast_table["condition"] == c) &
                (forecast_table["house"] == house) &
                (forecast_table["model"] == model_name)
            ]["NRMSE"].values[0]
            for c in conditions
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=model_name, color=COLORS[model_name])
        for xi, v in zip(x + (i - 1.5) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel("NRMSE (daylight hours only)", fontsize=11)
    ax.set_title(f"Forecast Accuracy by Condition — Household {house}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {outfile}")


def plot_economic_bars(control_table: pd.DataFrame, house: str, outfile: str):
    conditions = list(CONDITIONS.keys())
    scenario_map = {"arima": "SARIMA", "tft_prob": "TFT", "tft_residual": "TFT+Residual"}
    colors_econ = {"SARIMA": "#E24A33", "TFT": "#348ABD", "TFT+Residual": "#1B7837"}
    x = np.arange(len(conditions))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, (scenario_key, label) in enumerate(scenario_map.items()):
        rev_vals = [
            control_table[(control_table["condition"] == c) & (control_table["house"] == house) &
                           (control_table["scenario"] == scenario_key)]["revenue"].values[0]
            for c in conditions
        ]
        curt_vals = [
            control_table[(control_table["condition"] == c) & (control_table["house"] == house) &
                           (control_table["scenario"] == scenario_key)]["curtailment_kwh"].values[0]
            for c in conditions
        ]
        axes[0].bar(x + (i - 1) * width, rev_vals, width, label=label, color=colors_econ[label])
        axes[1].bar(x + (i - 1) * width, curt_vals, width, label=label, color=colors_econ[label])

    axes[0].set_xticks(x); axes[0].set_xticklabels(conditions)
    axes[0].set_ylabel("Net revenue (£/year)")
    axes[0].set_title("Economic outcome")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(x); axes[1].set_xticklabels(conditions)
    axes[1].set_ylabel("Curtailment (kWh/year)")
    axes[1].set_title("Wasted generation")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Battery Scheduling Outcomes — Household {house}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {outfile}")


if __name__ == "__main__":
    out_dir = file_config.results_dir

    forecast_table = build_forecast_table()
    forecast_table.to_csv(f"{out_dir}/presentation_forecast_metrics.csv", index=False)
    print("\n=== Forecast accuracy (NRMSE / MAE / RMSE) ===")
    print(forecast_table.round(4).to_string(index=False))

    prob_table = build_probabilistic_table()
    prob_table.to_csv(f"{out_dir}/presentation_probabilistic_metrics.csv", index=False)
    print("\n=== Probabilistic metrics (coverage / pinball / interval width) ===")
    print(prob_table.round(4).to_string(index=False))

    control_table = build_control_table()
    control_table.to_csv(f"{out_dir}/presentation_control_metrics.csv", index=False)
    print("\n=== Control / economic outcomes ===")
    print(control_table.round(3).to_string(index=False))

    for house in ["1", "2"]:
        plot_nrmse_bars(forecast_table, house, f"{out_dir}/presentation_nrmse_h{house}.png")
        plot_economic_bars(control_table, house, f"{out_dir}/presentation_economics_h{house}.png")

    print(f"\nAll tables + charts saved to {out_dir}/ (presentation_*.csv / presentation_*.png)")
