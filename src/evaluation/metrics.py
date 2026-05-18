import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import logging # type: ignore[import]
import wandb# type: ignore[import]
import argparse
from src.config import file_config, model_config # type: ignore[import]
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
    range_y = np.max(y_true) - np.min(y_true)
    return rmse_value / range_y if range_y != 0 else float('inf')

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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics (RMSE, NRMSE, MAE, MBE) for the given true and predicted values.
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted target values.
    Returns:
        dict: A dictionary containing all computed metrics."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MBE": mbe(y_true, y_pred)
    }

def main():
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
        training_dataset = create_dataset(train_data, model_config.max_encoder_length, model_config.max_prediction_length)
        test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
        test_dataloader = dataloader(test_dataset, batch_size=model_config.test_batch_size, train=False)
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