# mypy: ignore-errors
import torch # type: ignore[import]
import pandas as pd# type: ignore[import]
import numpy as np# type: ignore[import]
import logging
import sys
import wandb # type: ignore
from statsmodels.tsa.arima.model import ARIMA
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



def plot_predictions(results: pd.DataFrame, naive_pred: np.ndarray, name: str, corrector_type: str, hours: int = 168):
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
        name=f"{corrector_type} 80% interval", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=x, y=r["corrected_median"].values,
        name=f"TFT + {corrector_type}", line=dict(color="red", width=1.5)
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

def arima_baseline(raw_df: pd.DataFrame, results_df: pd.DataFrame, test_start: str = "2023-10-01") -> pd.DataFrame:
    # use raw df for pre-October history
    train = raw_df[raw_df["time"] < pd.Timestamp(test_start)]["P_norm"].values
    history = list(train)
    test_dates = sorted(results_df["date"].unique())
    all_results = []

    for test_day in test_dates:
        logging.info(f"ARIMA processing {test_day.date()}")
        model = ARIMA(history, order=(24, 0, 1))
        fit = model.fit()
        forecast = np.clip(fit.forecast(steps=24), 0, None)

        actual = results_df[results_df["date"] == test_day]["actual"].values

        if len(actual) != 24:
            continue

        history.extend(actual.tolist())
        all_results.append(pd.DataFrame({
            "actual": actual,
            "arima_pred": forecast,
            "date": test_day
        }))

    return pd.concat(all_results, ignore_index=True)
def ensemble_residuals(predictions: dict, window_sizes:list)-> pd.DataFrame:
    """
    group each residual with different sizes and average their corrections to achieve ensemble correction

    args:
        predictions (dict): results including correction for all window sizes
        window sizes (list): window sizes used
    
    returns:
        results(pd.DataFrame): contains results and tft predictrions and ground truth
    """

    corrections = np.array([predictions[w]["correction"].values for w in window_sizes])
    tft_median = predictions[window_sizes[0]]["tft_pred"].values
    tft_upper = predictions[window_sizes[0]]["tft_upper"].values
    tft_lower = predictions[window_sizes[0]]["tft_lower"].values
    actual = predictions[window_sizes[0]]["actual"].values
    daylight = predictions[window_sizes[0]]["daylight"].values
    
    correction = np.mean(corrections, axis=0)

    corrected_median = tft_median + correction
    corrected_upper = tft_upper + correction
    corrected_lower = tft_lower+correction

    results = pd.DataFrame({
        "actual": actual,
        "tft_pred": tft_median,
        "tft_lower": tft_lower,
        "tft_upper": tft_upper,
        "corrected_median": corrected_median,
        "corrected_upper": corrected_upper,
        "corrected_lower": corrected_lower,
        "correction": correction,
        "daylight": daylight,
        "date": predictions[window_sizes[0]]["date"].values,
        
    })
    results["corrected_median"] = results["corrected_median"].clip(lower=0)
    results["corrected_upper"] = results["corrected_upper"].clip(lower=0)
    results["corrected_lower"] = results["corrected_lower"].clip(lower=0)
    return results

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
    results_per_window = {}
    window_mae = {}
    window_rmse = {}
    for window in residuals_config.window_size:
        logging.info(f"running window size {window}")
        results = rolling_window_evaluation(df, predictions_df, window_days=window, epochs=50)
        results_per_window[window] = results
        corrected_metrics = compute_metrics(results, "corrected_median")
        window_mae[window] = corrected_metrics["MAE"]
        window_rmse[window] = corrected_metrics["RMSE"]
        results.to_csv(f"{file_config.results_dir}/{rolling_file}_{window}.csv", index=False)
        
        wandb.log({
            f"window_{window}_mae": corrected_metrics["MAE"],
            f"window_{window}_rmse": corrected_metrics["RMSE"],
        })
        logging.info(f"window_{window} - MAE: {corrected_metrics['MAE']:.4f}, RMSE: {corrected_metrics['RMSE']:.4f}")

    ensemble_results = ensemble_residuals(results_per_window, residuals_config.window_size)
    ensemble_metrics = compute_metrics(ensemble_results, "corrected_median")
    ensemble_results.to_csv(f"{file_config.results_dir}/{rolling_file}_ensemble.csv", index= False)
    tft_metrics = compute_metrics(results, "tft_pred")

    logging.info(f"ensemble MAE: {ensemble_metrics['MAE']:.4f}")
    logging.info(f"ensemble RMSE: {ensemble_metrics['RMSE']:.4f}")
    results["naive"] = naive_baseline(results)
    naive_metrics = compute_metrics(results, "naive")

    arima_results = arima_baseline(df, ensemble_results)
    arima_daytime = arima_results[arima_results["date"].isin(
        ensemble_results[ensemble_results["daylight"] == 1]["date"]
    )] if "daylight" in arima_results.columns else arima_results
    arima_metrics = compute_metrics(arima_results,"arima_pred")
    logging.info(f"ARIMA — MAE: {arima_metrics['MAE']:.4f}, RMSE: {arima_metrics['RMSE']:.4f}")
    wandb.log({"arima_mae": arima_metrics["MAE"], "arima_rmse": arima_metrics["RMSE"]})
    wandb.log({
        "tft_mae": tft_metrics["MAE"],
        "tft_rmse": tft_metrics["RMSE"],
        "naive_mae": naive_metrics["MAE"],
        "naive_rmse": naive_metrics["RMSE"],
        "ensemble_mae": ensemble_metrics["MAE"],
        "ensemble_rmse": ensemble_metrics["RMSE"],
    })

    logging.info(f"Naive     — MAE: {naive_metrics['MAE']:.4f}, RMSE: {naive_metrics['RMSE']:.4f}")
    logging.info(f"TFT       — MAE: {tft_metrics['MAE']:.4f}, RMSE: {tft_metrics['RMSE']:.4f}")
    


    
    plot_predictions(ensemble_results, results["naive"], "forecast_oct_week", "ensemble residual", hours=168)
    plot_predictions(ensemble_results.iloc[1000:], results["naive"][1000:], "forecast_dec_week","ensemble_residual", hours=168)
    
    wandb.finish()
    logging.info("Evaluation complete")