# mypy: ignore-errors
import torch # type: ignore[import]
import pandas as pd# type: ignore[import]
import numpy as np# type: ignore[import]
import logging
import sys
import wandb # type: ignore
import plotly.graph_objects as go
import matplotlib.pyplot as plt # type: ignore[import]
from src.config import file_config, residuals_config
from src.utils.utils import wandb_login
from src.models.residual_corrector import ResidualCorrector
from src.models.train_residuals import rolling_window_evaluation
from src.evaluation.metrics import compute_metrics
wandb.init = wandb.init  # type: ignore

logging.basicConfig(level=logging.INFO)

def load_residual_model() -> ResidualCorrector:
    model = ResidualCorrector(
        input_size=11,
        hidden_size=residuals_config.hidden_size
    )
    model.load_state_dict(torch.load(f"{file_config.models_dir}/residual/residual_corrector.pth"))
    model.eval()
    return model



def plot_predictions(results: pd.DataFrame, naive_pred: np.ndarray, name: str, hours: int = 168):
    """Plot actual vs TFT vs TFT+Residual with uncertainty bands using plotly."""
    r = results.iloc[:hours].copy()
    naive = naive_pred[:hours]
    x = list(range(hours))

    fig = go.Figure()

    # actual
    fig.add_trace(go.Scatter(
        x=x, y=r["actual"].values,
        name="Actual", line=dict(color="black", width=2)
    ))

    # TFT uncertainty band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(r["tft_upper"].values) + list(r["tft_lower"].values[::-1]),
        fill="toself", fillcolor="rgba(0,0,255,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="TFT 80% interval", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=x, y=r["tft_pred"].values,
        name="TFT Median", line=dict(color="blue", width=1.5)
    ))

    # TFT+Residual uncertainty band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(r["corrected_upper"].values) + list(r["corrected_lower"].values[::-1]),
        fill="toself", fillcolor="rgba(255,0,0,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Corrected 80% interval", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=x, y=r["corrected_median"].values,
        name="TFT + Residual", line=dict(color="red", width=1.5)
    ))

    # naive
    fig.add_trace(go.Scatter(
        x=x, y=naive,
        name="Naive Baseline", line=dict(color="green", width=1, dash="dash")
    ))

    fig.update_layout(
        title="Solar PV Forecast: Actual vs TFT vs TFT+Residual Corrector",
        xaxis_title="Time (hours)",
        yaxis_title="Normalised Power Output",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        height=400
    )

    wandb.log({name: fig})


def naive_baseline(results: pd.DataFrame) -> np.ndarray:
    """return persistent naive baseline"""
    return results["actual"].shift(24).fillna(0).values


if __name__ == "__main__":

    wandb_login()
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "finetuned"
    
    if mode == "finetuned":
        run_name = "residual-evaluation-finetuned"
        predictions_file = "predictions_ukpv_finetuned.csv"
        rolling_file = "predictions_rolling_finetuned.csv"
        metrics_prefix = "finetuned"
    else:
        run_name = "residual-evaluation-pretrained"
        predictions_file = "predictions_ukpv_pretrained.csv"
        rolling_file = "predictions_rolling_pretrained.csv"
        metrics_prefix = "pretrained"

    wandb.init(project="pv-forecasting", name=run_name)
    
    df = pd.read_parquet(file_config.test_set)
    predictions_df = pd.read_csv(f"{file_config.results_dir}/{predictions_file}")
    
    results = rolling_window_evaluation(df, predictions_df, window_days=30, epochs=50)
    results.to_csv(f"{file_config.results_dir}/{rolling_file}", index=False)
    
    tft_metrics = compute_metrics(results["actual"].values, results["tft_pred"].values)
    corrected_metrics = compute_metrics(results["actual"].values, results["corrected_median"].values)
    naive_pred = naive_baseline(results)
    naive_metrics = compute_metrics(results["actual"].values, naive_pred)
    
    wandb.log({
        "tft_mae": tft_metrics["MAE"],
        "tft_rmse": tft_metrics["RMSE"],
        "corrected_mae": corrected_metrics["MAE"],
        "corrected_rmse": corrected_metrics["RMSE"],
        "naive_mae": naive_metrics["MAE"],
        "naive_rmse": naive_metrics["RMSE"],
        "mae_improvement": tft_metrics["MAE"] - corrected_metrics["MAE"],
        "rmse_improvement": tft_metrics["RMSE"] - corrected_metrics["RMSE"],
    })
    
    logging.info(f"Naive     — MAE: {naive_metrics['MAE']:.4f}, RMSE: {naive_metrics['RMSE']:.4f}")
    logging.info(f"TFT       — MAE: {tft_metrics['MAE']:.4f}, RMSE: {tft_metrics['RMSE']:.4f}")
    logging.info(f"Corrected — MAE: {corrected_metrics['MAE']:.4f}, RMSE: {corrected_metrics['RMSE']:.4f}")
    
    pd.DataFrame([naive_metrics]).to_csv(f"{file_config.results_dir}/metrics_naive.csv", index=False)
    pd.DataFrame([tft_metrics]).to_csv(f"{file_config.results_dir}/metrics_tft_{metrics_prefix}.csv", index=False)
    pd.DataFrame([corrected_metrics]).to_csv(f"{file_config.results_dir}/metrics_residual_{metrics_prefix}.csv", index=False)
    
    plot_predictions(results, naive_pred, "forecast_oct_week", hours=168)
    plot_predictions(results.iloc[1000:], naive_pred[1000:], "forecast_dec_week", hours=168)
    
    wandb.finish()
    logging.info("Evaluation complete")