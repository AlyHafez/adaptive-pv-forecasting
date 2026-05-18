# mypy: ignore-errors
import torch # type: ignore[import]
import pandas as pd# type: ignore[import]
import numpy as np# type: ignore[import]
import logging
import wandb # type: ignore
import matplotlib.pyplot as plt # type: ignore[import]
from src.config import file_config, residuals_config
from src.utils.utils import wandb_login
from src.models.residual_corrector import ResidualCorrector
from src.models.train_residuals import create_dataset, dataloader
from src.evaluation.metrics import compute_metrics
wandb.init = wandb.init  # type: ignore

logging.basicConfig(level=logging.INFO)

def load_residual_model() -> ResidualCorrector:
    model = ResidualCorrector(
        input_size=7,
        hidden_size=residuals_config.hidden_size
    )
    model.load_state_dict(torch.load(f"{file_config.models_dir}/residual/residual_corrector.pth"))
    model.eval()
    return model

def evaluate_residuals(
    test_df: pd.DataFrame,
    test_predictions: np.ndarray,
    model: ResidualCorrector
) -> pd.DataFrame:
    """Evaluate TFT + residual corrector on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    features = test_df[["hour_sin", "hour_cos", "month_sin", "month_cos", 
                         "dayofyear_sin", "dayofyear_cos"]].values
    
    if test_predictions.ndim == 3:
        pred_median = test_predictions[:, :, 1].flatten()
    else:
        pred_median = test_predictions.flatten()
    
    X = np.column_stack([pred_median, features])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        correction = model(X_tensor).squeeze().cpu().numpy()
    
    corrected = pred_median + correction
    actuals = test_df["P_norm"].values
    
    results = pd.DataFrame({
        "actual": actuals,
        "tft_pred": pred_median,
        "corrected_pred": corrected,
        "correction": correction,
    })
    return results


def plot_predictions(results: pd.DataFrame, save_path: str, naive_pred: np.ndarray):
    """Plot actual vs TFT vs TFT+Residual predictions."""
    fig, ax = plt.subplots(figsize=(15, 5))
    
    ax.plot(results["actual"].values, label="Actual", color="black", linewidth=1.5)
    ax.plot(results["tft_pred"].values, label="TFT", color="blue", linewidth=1, alpha=0.7)
    ax.plot(results["corrected_pred"].values, label="TFT + Residual", color="red", linewidth=1, alpha=0.7)
    ax.plot(naive_pred, label="Naive Baseline", color="green", linewidth=1, alpha=0.5, linestyle="--")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Normalised Power Output")
    ax.set_title("Solar PV Forecast: Actual vs TFT vs TFT+Residual Corrector")
    
    

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    wandb.log({"predictions_plot": wandb.Image(fig)}) 
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}")


def naive_baseline(test_df: pd.DataFrame) -> np.ndarray:
    """Persistence baseline — repeat same hour from 24h ago."""
    return test_df["P_norm"].shift(24).fillna(0).values


if __name__ == "__main__":
    wandb_login()
    wandb.init(project="pv-forecasting", name="residual-evaluation")
    
    df = pd.read_parquet(f"{file_config.processed_data_dir}/ukpv_london_tft.parquet")
    last_date = df["time"].max()
    test_df = df[df["time"] >= last_date - pd.Timedelta(days=7)]
    
    predictions_df = pd.read_csv(f"{file_config.results_dir}/predictions_ukpv.csv")
    # take last 7 days of predictions
    test_predictions = predictions_df["median"].values[-len(test_df):]
    
    model = load_residual_model()
    results = evaluate_residuals(test_df, test_predictions, model)
    tft_metrics = compute_metrics(results["actual"].values, results["tft_pred"].values)
    corrected_metrics = compute_metrics(results["actual"].values, results["corrected_pred"].values)
    
    wandb.log({
        "tft_mae": tft_metrics["MAE"],
        "tft_rmse": tft_metrics["RMSE"],
        "corrected_mae": corrected_metrics["MAE"],
        "corrected_rmse": corrected_metrics["RMSE"],
        "mae_improvement": tft_metrics["MAE"] - corrected_metrics["MAE"],
        "rmse_improvement": tft_metrics["RMSE"] - corrected_metrics["RMSE"],
    })
    logging.info(f"TFT — MAE: {tft_metrics['MAE']:.4f}, RMSE: {tft_metrics['RMSE']:.4f}")
    logging.info(f"Corrected — MAE: {corrected_metrics['MAE']:.4f}, RMSE: {corrected_metrics['RMSE']:.4f}")
    logging.info(f"Improvement — MAE: {tft_metrics['MAE'] - corrected_metrics['MAE']:.4f}, RMSE: {tft_metrics['RMSE'] - corrected_metrics['RMSE']:.4f}")
    
    pd.DataFrame([tft_metrics]).to_csv(f"{file_config.results_dir}/metrics_tft_ukpv.csv", index=False)
    pd.DataFrame([corrected_metrics]).to_csv(f"{file_config.results_dir}/metrics_residual_corrector.csv", index=False)
    results.to_csv(f"{file_config.results_dir}/predictions_residual_corrector.csv", index=False)
    

    naive_pred = naive_baseline(test_df)
    naive_metrics = compute_metrics(results["actual"].values, naive_pred)
    plot_predictions(results, f"{file_config.results_dir}/predictions_plot.png", naive_pred=naive_pred)

    wandb.log({
        "naive_mae": naive_metrics["MAE"],
        "naive_rmse": naive_metrics["RMSE"],
    })
    logging.info(f"Naive — MAE: {naive_metrics['MAE']:.4f}, RMSE: {naive_metrics['RMSE']:.4f}")
    pd.DataFrame([naive_metrics]).to_csv(f"{file_config.results_dir}/metrics_naive.csv", index=False)
    wandb.finish()
    logging.info("Evaluation complete")