import pandas as pd # type: ignore[import]
import logging
from pytorch_forecasting import TimeSeriesDataSet # type: ignore[import]
from pytorch_forecasting.data import TorchNormalizer # type: ignore[import]
from src.config import file_config, tft_config 
logging.basicConfig(level=logging.INFO)
def split_data(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_data = data[data["time"].dt.year <= 2021]
    val_data = data[data["time"].dt.year == 2022]
    test_data = data[data["time"].dt.year == 2023]

    return train_data, val_data, test_data
def split_household_data(
        data:pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = data[data["time"].dt.month <= 10].reset_index(drop=True)
    val_data = data[data["time"].dt.month > 10].reset_index(drop=True)
    return train_data, val_data

def create_dataset(data:pd.DataFrame, max_encoder_length:int, max_prediction_length:int) -> TimeSeriesDataSet:
    """Creare a TimeSeriesDataSet for training the Temporal Fusion Transformer model.
    Args:
        data (pd.DataFrame): The input DataFrame containing the time series data with features.
        max_encoder_length (int): The maximum length of the encoder input sequence.
        max_prediction_length (int): The maximum length of the prediction output sequence.
        Returns:
        TimeSeriesDataSet: A PyTorch Forecasting TimeSeriesDataSet ready for model training."""
    training = TimeSeriesDataSet(
        data, # the input DataFrame with features
        time_idx = "time_idx", # column representing the time index per location
        target = "P_norm",# the target variable to predict (normalized power output)
        group_ids=["series_id"],# column(s) that identify different time series (locations in this case)
        max_encoder_length = max_encoder_length, # the length of the input sequence for the encoder
        max_prediction_length = max_prediction_length, # the length of the output sequence to predict
        static_categoricals = ["location"], # categorical features that do not change over time (location names)
        static_reals = ["lat", "lon"], # real-valued features that do not change over time (latitude and longitude)
        time_varying_known_reals = [
            "hour_sin", "hour_cos", "month_sin", "month_cos", 
            "dayofyear_sin", "dayofyear_cos", "Gb(i)", "Gd(i)",   
            "Gr(i)", "H_sun","T2m", "WS10m", "daylight"  
            ],# real-valued features that are known in advance and vary over time (time-based features and weather conditions)
        time_varying_unknown_reals = ["P_norm"], # real-valued features that are not known in advance and vary over time (the target variable)
        lags={
        "P_norm": [24, 168]
        }, # specify lag features for the target variable (e.g., 24 hours and 168 hours for daily and weekly patterns)
        target_normalizer = TorchNormalizer(method="identity", center=False),
        add_target_scales=False,
        add_encoder_length="auto"
    )
    return training

def dataloader(dataset: TimeSeriesDataSet, batch_size:int, train:bool=True, num_workers:int=8) -> TimeSeriesDataSet.to_dataloader:
    """Create a dataloader from a TimeSeriesDataSet for model training or evaluation.
    Args:
        dataset (TimeSeriesDataSet): The input TimeSeriesDataSet to create the dataloader from.
        batch_size (int): The batch size to use for the dataloader.
        train (bool): Whether to create a dataloader for training (with shuffling) or evaluation (without shuffling).
        num_workers (int): The number of worker processes to use for data loading.
    Returns:
        DataLoader: A PyTorch DataLoader that can be used for model training or evaluation."""
    return dataset.to_dataloader(
        train=train,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True
    )

if __name__ == "__main__":
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, test_data = split_data(data)
    logging.info(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    max_encoder_length = tft_config.max_encoder_length# use the past week of data
    max_prediction_length = tft_config.max_prediction_length # predict the next 24 hours
    training_dataset = create_dataset(train_data, max_encoder_length, max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
    logging.info(f"Training dataset: {len(training_dataset)}, Validation dataset: {len(validation_dataset)}, Test dataset: {len(test_dataset)}")

    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=tft_config.train_batch_size,
        num_workers=0
    )

    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=tft_config.val_batch_size,
        num_workers=0
    )

    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=tft_config.test_batch_size,
        num_workers=0
    )
    logging.info(f"Created dataloaders with sizes - Train: {len(train_dataloader)}, Val: {len(val_dataloader)}, Test: {len(test_dataloader)}")
    x, y = next(iter(train_dataloader))
    logging.info(f"Example batch - x shape: {x['encoder_cont'].shape}, y shape: {y[0].shape}")
  