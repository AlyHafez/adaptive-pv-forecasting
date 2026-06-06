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



def plot_predictions(results: pd.DataFrame, naive_pred: np.ndarray, arima_baseline: np.ndarray, name: str, corrector_type: str, hours: int = 168):
    """Plot actual vs TFT vs TFT+Residual with uncertainty bands using plotly."""
    r = results.iloc[:hours].copy()
    naive = naive_pred[:hours]
    arima = arima_baseline[:hours]
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
    # arima
    fig.add_trace(go.Scatter(
        x=x, y=arima,
        name="ARIMA Baseline", line=dict(color="orange", width=1, dash="dash")
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

def arima_baseline(df, ensemble_results, window_days=30):
    predictions = []
    dates = []
    daylights = []
    
    for test_day in pd.date_range("2023-01-01", "2023-12-31", freq="D"):
        window_start = test_day - pd.Timedelta(days=window_days)
        
        history = df[
            (df["time"] >= window_start) & 
            (df["time"] < test_day)
        ]["P_norm"].values
        
        # fix - skip if not enough history
        if len(history) < 48:  # need at least 2 days
            logging.warning(f"Insufficient history for ARIMA on {test_day.date()}, skipping")
            # append NaN placeholder so indices stay aligned
            test_hours = df[df["time"].dt.date == test_day.date()]
            predictions.extend([np.nan] * len(test_hours))
            dates.extend(test_hours["time"].values)
            daylights.extend(test_hours["daylight"].values)
            continue
        
        logging.info(f"ARIMA processing {test_day.date()}")
        
        try:
            model = ARIMA(history, order=(2, 0, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=24)
            forecast = np.clip(forecast, 0, None)
        except Exception as e:
            logging.warning(f"ARIMA failed on {test_day.date()}: {e}")
            forecast = np.zeros(24)
        
        test_hours = df[df["time"].dt.date == test_day.date()]
        predictions.extend(forecast[:len(test_hours)])
        dates.extend(test_hours["time"].values)
        daylights.extend(test_hours["daylight"].values)
    
    return np.array(predictions), np.array(dates), np.array(daylights)

def ensemble_residuals(predictions: dict, window_sizes:list, method:str)-> pd.DataFrame:
    """
    group each residual with different sizes and average their corrections to achieve ensemble correction

    args:
        predictions (dict): results including correction for all window sizes
        window sizes (list): window sizes used
    
    returns:
        results(pd.DataFrame): contains results and tft predictrions and ground truth
    """
    # normalise dates to date only before comparing
    for w in window_sizes:
        predictions[w] = predictions[w].copy()
        predictions[w]["date"] = pd.to_datetime(predictions[w]["date"]).dt.date

    common_dates = set(predictions[window_sizes[0]]["date"])
    for w in window_sizes[1:]:
        common_dates &= set(predictions[w]["date"])

    logging.info(f"Common dates across all windows: {len(common_dates)} days")
    logging.info(f"Window lengths before filter: {[len(predictions[w]) for w in window_sizes]}")

    predictions = {
        w: predictions[w][
            predictions[w]["date"].isin(common_dates)
        ].reset_index(drop=True)
        for w in window_sizes
    }

    logging.info(f"Window lengths after filter: {[len(predictions[w]) for w in window_sizes]}")

    # verify alignment before proceeding
    lengths = [len(predictions[w]) for w in window_sizes]
    assert len(set(lengths)) == 1, f"Still misaligned after filtering: {dict(zip(window_sizes, lengths))}"
    
    corrections = np.array([predictions[w]["correction"].values for w in window_sizes])
    tft_median = predictions[window_sizes[0]]["tft_pred"].values
    tft_upper = predictions[window_sizes[0]]["tft_upper"].values
    tft_lower = predictions[window_sizes[0]]["tft_lower"].values
    actual = predictions[window_sizes[0]]["actual"].values
    daylight = predictions[window_sizes[0]]["daylight"].values
    if method == "inverse_rmse":
        rmses = np.array([compute_metrics(predictions[win],
                         "corrected_median")["RMSE"]
                         for win in window_sizes])
        w = (1 / rmses) / (1 / rmses).sum()



    else: 
        w = np.ones(len(window_sizes)) / len(window_sizes)

    correction = np.average(corrections, weights=w, axis=0)

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
        "time": predictions[window_sizes[0]]["time"].values,
        
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
        ensemble_file = "predictions_rolling_finetuned_ensemble.csv"
        truth_path = file_config.test_set
        metrics_prefix = "finetuned"
    elif mode == "pretrained":
        run_name = "residual-evaluation-pretrained"
        predictions_file = "predictions_ukpv_pretrained.csv"
        rolling_file = "predictions_rolling_pretrained.csv"
        ensemble_file = "predictions_rolling_pretrained_ensemble.csv"
        truth_path = file_config.test_set
        metrics_prefix = "pretrained"
    elif mode == "finetuned_drifted":
        run_name = "residual-evaluation-finetuned-drifted"
        predictions_file = "predictions_ukpv_finetuned_drifted.csv"
        rolling_file = "predictions_rolling_finetuned_drifted.csv"
        ensemble_file = "predictions_rolling_finetuned_drifted_ensemble.csv"
        truth_path = file_config.test_set_drifted
        metrics_prefix = "finetuned_drifted"
    elif mode == "pretrained_drifted":
        run_name = "residual-evaluation-pretrained-drifted"
        predictions_file = "predictions_ukpv_pretrained_drifted.csv"
        rolling_file = "predictions_rolling_pretrained_drifted.csv"
        ensemble_file = "predictions_rolling_pretrained_drifted_ensemble.csv"
        truth_path = file_config.test_set_drifted
        metrics_prefix = "pretrained_drifted"
    elif mode == "finetuned_shaded":
        run_name = "residual-evaluation-finetuned-shaded"
        predictions_file = "predictions_ukpv_finetuned_shaded.csv"
        rolling_file = "predictions_rolling_finetuned_shaded.csv"
        ensemble_file = "predictions_rolling_finetuned_shaded_ensemble.csv"
        truth_path = file_config.test_set_shaded
        metrics_prefix = "finetuned_shaded"
    elif mode == "pretrained_shaded":
        run_name = "residual-evaluation-pretrained-shaded"
        predictions_file = "predictions_ukpv_pretrained_shaded.csv"
        rolling_file = "predictions_rolling_pretrained_shaded.csv"
        ensemble_file = "predictions_rolling_pretrained_shaded_ensemble.csv"
        truth_path = file_config.test_set_shaded
        metrics_prefix = "pretrained_shaded"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    wandb.init(project="pv-forecasting", name=run_name)
    
    df = pd.read_parquet(truth_path)
    predictions_df = pd.read_csv(f"{file_config.results_dir}/{predictions_file}")
    logging.info(f"predictions_df shape: {predictions_df.shape}")
    logging.info(f"predictions_df time sample: {predictions_df['time'].head(5).values}")
    logging.info(f"df shape: {df.shape}")
  
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

    ensemble_results = ensemble_residuals(results_per_window, residuals_config.window_size, "inverse_rmse")
    ensemble_metrics = compute_metrics(ensemble_results, "corrected_median")

    tft_metrics = compute_metrics(ensemble_results, "tft_pred")

    logging.info(f"ensemble MAE: {ensemble_metrics['MAE']:.4f}")
    logging.info(f"ensemble RMSE: {ensemble_metrics['RMSE']:.4f}")
    ensemble_results["naive"] = naive_baseline(ensemble_results)
    naive_metrics = compute_metrics(ensemble_results, "naive")

    arima_pred, arima_dates, arima_daylights = arima_baseline(df, ensemble_results)

    # create temp df and merge
    arima_df = pd.DataFrame({
        "time": pd.to_datetime(arima_dates),
        "arima_pred": arima_pred,
    })

    ensemble_results = ensemble_results.merge(
        arima_df, on="time", how="left"
    )
    arima_daytime = ensemble_results[ensemble_results["daylight"] == 1]
    arima_metrics = compute_metrics(arima_daytime, "arima_pred")
    logging.info(f"ARIMA — MAE: {arima_metrics['MAE']:.4f}, RMSE: {arima_metrics['RMSE']:.4f}")
    wandb.log({"arima_mae": arima_metrics["MAE"], "arima_rmse": arima_metrics["RMSE"]})
    wandb.log({
        "arima_mae": arima_metrics["MAE"],
        "arima_rmse": arima_metrics["RMSE"],
        "arima_nrmse": arima_metrics["NRMSE"],
        "tft_mae": tft_metrics["MAE"],
        "tft_rmse": tft_metrics["RMSE"],
        "tft_nrmse": tft_metrics["NRMSE"],
        "naive_mae": naive_metrics["MAE"],
        "naive_rmse": naive_metrics["RMSE"],
        "naive_nrmse": naive_metrics["NRMSE"],
        "ensemble_mae": ensemble_metrics["MAE"],
        "ensemble_rmse": ensemble_metrics["RMSE"],
        "ensemble_nrmse": ensemble_metrics["NRMSE"],
    })

    logging.info(f"Naive     — MAE: {naive_metrics['MAE']:.4f}, RMSE: {naive_metrics['RMSE']:.4f}, NRMSE: {naive_metrics['NRMSE']:.4f}")
    logging.info(f"TFT       — MAE: {tft_metrics['MAE']:.4f}, RMSE: {tft_metrics['RMSE']:.4f}, NRMSE: {tft_metrics['NRMSE']:.4f}")
    
    
    ensemble_results.to_csv((f"{file_config.results_dir}/{ensemble_file}"), index=False)
    logging.info("\n=== Final Metrics Summary ===")
    logging.info(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'NRMSE':>8}")
    logging.info(f"{'Naive':<20} {naive_metrics['MAE']:>8.4f} {naive_metrics['RMSE']:>8.4f} {naive_metrics['NRMSE']:>8.4f}")
    logging.info(f"{'ARIMA':<20} {arima_metrics['MAE']:>8.4f} {arima_metrics['RMSE']:>8.4f} {arima_metrics['NRMSE']:>8.4f}")
    logging.info(f"{'TFT':<20} {tft_metrics['MAE']:>8.4f} {tft_metrics['RMSE']:>8.4f} {tft_metrics['NRMSE']:>8.4f}")
    logging.info(f"{'TFT+Residual':<20} {ensemble_metrics['MAE']:>8.4f} {ensemble_metrics['RMSE']:>8.4f} {ensemble_metrics['NRMSE']:>8.4f}")
    plot_predictions(ensemble_results, ensemble_results["naive"], ensemble_results["arima_pred"], "forecast_week", "ensemble residual", hours=168)
    plot_predictions(
        ensemble_results.iloc[6000:],
        ensemble_results["naive"].iloc[6000:].values,
        ensemble_results["arima_pred"].iloc[6000:].values,
        "forecast_sep_week", "ensemble_residual", hours=168
    )
    
    wandb.finish()
    logging.info("Evaluation complete")