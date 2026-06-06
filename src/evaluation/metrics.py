import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import logging # type: ignore[import]
import wandb# type: ignore[import]
import argparse
from src.config import file_config, tft_config # type: ignore[import]
from src.utils.utils import wandb_login # type: ignore[import]
from src.models.eval import load_model, evaluate_tft # type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader
from pytorch_forecasting import TimeSeriesDataSet # type: ignore[import]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error (RMSE) between true and predicted values.
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
    Returns:
        float: The calculated RMSE value."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
    Returns:
        float: The calculated NRMSE value."""
    rmse_value = rmse(y_true, y_pred)
    mean_y = np.mean(y_true)
    return rmse_value / mean_y if mean_y != 0 else float('inf')

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error (MAE) between true and predicted values.
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
    Returns:
        float: The calculated MAE value."""
    return np.mean(np.abs(y_true - y_pred))

def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Bias Error (MBE) between true and predicted values.
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
    Returns:
        float: The calculated MBE value."""
    return np.mean(y_pred - y_true)
def pinball_loss(actual, predicted, quantile):
    """
    Pinball loss for a single quantile.
    quantile: 0.1 for Q10, 0.5 for Q50, 0.9 for Q90
    """
    errors = actual - predicted
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))
def compute_probabilistic(results:pd.DataFrame, target:str) -> dict:
    """Calculate the Pinball Loss for quantile predictions.
    Args:
        results (pd.DataFrame): A DataFrame containing the true values and quantile predictions.
        target (str): The column name of the true target values in the DataFrame.
    Returns:
        float: The calculated Pinball Loss value."""
    daylight_only = results[results["daylight"] == 1].reset_index(drop=True)
    actual = daylight_only["actual"].values
    if target == "tft_pred":
        lower = daylight_only["tft_lower"].values
        median = daylight_only["tft_pred"].values
        upper = daylight_only["tft_upper"].values
    elif target == "corrected_median":
        lower = daylight_only["corrected_lower"].values
        median = daylight_only["corrected_median"].values
        upper = daylight_only["corrected_upper"].values
    
    return {
        "pinball_q10": pinball_loss(actual, lower, 0.1),
        "pinball_q50": pinball_loss(actual, median, 0.5),
        "pinball_q90": pinball_loss(actual, upper, 0.9),
        "pinball_mean": np.mean([
            pinball_loss(actual, lower, 0.1),
            pinball_loss(actual, median, 0.5),
            pinball_loss(actual, upper, 0.9)
        ]),
        "coverage": np.mean((actual >= lower) & (actual <= upper)),
        "interval_width": np.mean(upper - lower),
    }
def compute_metrics(results:pd.DataFrame, target:str,) -> dict:
    """Compute all evaluation metrics (RMSE, NRMSE, MAE, MBE) for the given true and predicted values excluding night times.
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
        daylight (bool): flag identifying day and night times
    Returns:
        dict: A dictionary containing all computed metrics."""
    
    daylight_only_results = results[results["daylight"] == 1].reset_index(drop=True)
    y_true = daylight_only_results["actual"].values
    y_pred = daylight_only_results[target].values
    return {
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MBE": mbe(y_true, y_pred)
    }

def main():
    """Main function to evaluate the TFT model and compute metrics."""
    wandb_login()
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", action="store_true", help="Load predictions from CSV instead of running model")
    args = parser.parse_args()
    wandb.init(project="pv-forecasting", name="tft-evaluation")
    if args.from_csv:
        predictions_df = pd.read_csv(f"{file_config.results_dir}/predictions_tft.csv")
        y_true = predictions_df["actual"].values
        y_pred = predictions_df["median"].values
    else:

        data = pd.read_parquet(file_config.data_path)
        train_data, val_data, test_data = split_data(data)
        training_dataset = create_dataset(train_data, tft_config.max_encoder_length, tft_config.max_prediction_length)
        test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
        test_dataloader = dataloader(test_dataset, batch_size=tft_config.test_batch_size, train=False)
        checkpoint_path = f"{file_config.models_dir}/tft/tft-best-model.ckpt"  
        best_model = load_model(checkpoint_path)
        predictions = evaluate_tft(best_model, test_dataloader)
        y_true = predictions.y[0].numpy().flatten()
        y_pred = predictions.output[:, :, 1].numpy().flatten()  # median prediction

    metrics = compute_metrics(y_true, y_pred)
    pd.DataFrame([metrics]).to_csv(f"{file_config.results_dir}/metrics_tft.csv", index=False)
    logging.info(f"Metrics: {metrics}")
    wandb.log(metrics)
if __name__ == "__main__":
    main()