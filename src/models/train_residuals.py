# mypy: ignore-errors
import torch # type: ignore[import]
import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import torch.nn as nn # type: ignore[import]
import logging # type: ignore[import]
import wandb # type: ignore
import os
from src.models.eval_residual import evaluate_residuals
from src.config import file_config, residuals_config # type: ignore[import]
from src.utils.utils import wandb_login, set_seed # type: ignore[import]

from src.models.residual_corrector import ResidualCorrector # type: ignore[import]
logging.basicConfig(level=logging.INFO)
wandb_login()
def create_dataset(dataset: pd.DataFrame, predictions: np.ndarray) -> torch.utils.data.Dataset:
    """Create a PyTorch dataset for training the residuals model.
    Args:
        dataset (pd.DataFrame): The input dataset containing features and target variable.
        predictions (np.ndarray): The predicted values from the TFT model.
    Returns:
        torch.utils.data.Dataset: A PyTorch dataset ready for training."""
    features = dataset[[ "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"]].values
    
    if predictions.ndim == 3:
        pred_median = predictions[:, :, 1].flatten()  # from TFT directly
    else: pred_median = predictions.flatten()  # from saved CSV
    X = np.column_stack([pred_median, features])
    y = dataset["P_norm"].values - pred_median  # residuals (actual - predicted median)
    return torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

def dataloader(dataset: torch.utils.data.Dataset, batch_size: int, train: bool = True) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader from a dataset.
    Args:
        dataset (torch.utils.data.Dataset): The input dataset to create the DataLoader from.
        batch_size (int): The batch size to use for the DataLoader.
        train (bool): Whether to create a DataLoader for training (with shuffling) or evaluation (without shuffling).
    Returns:
        torch.utils.data.DataLoader: A PyTorch DataLoader ready for training or evaluation."""
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)


def train_residuals(
    train_df: pd.DataFrame,
    train_predictions: np.ndarray,
    epochs: int = 50
) -> ResidualCorrector:
    
    set_seed(residuals_config.seed)
    wandb.init(project="pv-forecasting", name="residual-corrector")
    
    dataset = create_dataset(train_df, train_predictions)
    train_loader = dataloader(dataset, batch_size=residuals_config.batch_size, train=True)
    
    model = ResidualCorrector(input_size=7, hidden_size=residuals_config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=residuals_config.lr)
    criterion = nn.MSELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})  # log to W&B
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{file_config.models_dir}/residual/residual_corrector.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} — Best loss: {best_loss:.6f}")
                break
    
    wandb.log({"best_loss": best_loss})
    wandb.finish()
    logging.info(f"Training complete — Best loss: {best_loss:.6f}")
    return model

def rolling_window_evaluation(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    window_days: int = 30,
    epochs: int = 50
) -> pd.DataFrame:
    """Rolling window evaluation — retrain MLP each day on last window_days of residuals."""
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_day in pd.date_range("2023-10-01", "2023-12-31", freq="D"):
        # get window before test day
        window_start = test_day - pd.Timedelta(days=window_days)
        window_df = df[
            (df["time"] >= window_start) & (df["time"] < test_day)
        ].reset_index(drop=True)

        if len(window_df) == 0:
            continue

        # get TFT predictions for window using positional index
        window_predictions = predictions_df[
            pd.to_datetime(predictions_df["time"]).dt.date.between(
                window_start.date(), (test_day - pd.Timedelta(hours=1)).date()
            )
        ]["median"].values
        if len(window_predictions) == 0:
            logging.warning(f"No predictions for window ending {test_day.date()}, skipping")
            continue

        # align lengths
        min_len = min(len(window_df), len(window_predictions))
        window_df = window_df.iloc[-min_len:].reset_index(drop=True)
        window_predictions = window_predictions[-min_len:]
        # retrain MLP on window — no wandb logging per iteration
        set_seed(residuals_config.seed)
        dataset = create_dataset(window_df, window_predictions)
        train_loader = dataloader(dataset, batch_size=residuals_config.batch_size, train=True)
        model = ResidualCorrector(input_size=7, hidden_size=residuals_config.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=residuals_config.lr)
        criterion = nn.MSELoss()
        model = model.to(device)
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch).squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        # predict next 24h
        test_mask = df["time"].dt.date == test_day.date()
        test_day_df = df[test_mask].reset_index(drop=True)
        test_preds = predictions_df[
            pd.to_datetime(predictions_df["time"]).dt.date == test_day.date()
        ]["median"].values

        if len(test_preds) == 0:
            logging.warning(f"No predictions for {test_day.date()}, skipping")
            continue

        if len(test_preds) != len(test_day_df):
            logging.warning(f"Incomplete predictions for {test_day.date()}: {len(test_preds)} vs {len(test_day_df)}, skipping")
            continue

        day_results = evaluate_residuals(test_day_df, test_preds, model)
        day_results["date"] = test_day
        all_results.append(day_results)
        logging.info(f"Processed {test_day.date()}")

    return pd.concat(all_results, ignore_index=True)
    


if __name__ == "__main__":

    
    
    df = pd.read_parquet(f"{file_config.test_set}")
    

    # In train_residuals.py __main__
    train_df = df[df["time"].dt.month <= 9] 

    
    predictions_df = pd.read_csv(f"{file_config.results_dir}/predictions_ukpv.csv")
    train_predictions = predictions_df["median"].values[:len(train_df)]

    logging.info(f"train_df: {train_df.shape}, predictions: {len(train_predictions)}")
    
    os.makedirs(f"{file_config.models_dir}/residual", exist_ok=True)
    model = train_residuals(train_df, train_predictions, epochs=50)