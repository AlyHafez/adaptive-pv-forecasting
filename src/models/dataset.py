import pandas as pd # type: ignore[import]
from pytorch_forecasting import TimeSeriesDataSet # type: ignore[import]
from pytorch_forecasting.data import GroupNormalizer # type: ignore[import]
from src.data.data_config import file_config # type: ignore[import]
def split_data(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_data = data[data["time"].dt.year <= 2021]
    val_data = data[data["time"].dt.year <= 2022]
    test_data = data[data["time"].dt.year <= 2023]

    return train_data, val_data, test_data

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
        group_ids=["location"],# column(s) that identify different time series (locations in this case)
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
        target_normalizer = GroupNormalizer(groups=["location"]), # normalize the target variable per location to help model training
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length="auto"
    )
    return training


if __name__ == "__main__":
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, test_data = split_data(data)
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    max_encoder_length = 168 # use the past week of data
    max_prediction_length = 24 # predict the next 24 hours
    training_dataset = create_dataset(train_data, max_encoder_length, max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
    print(f"Training dataset: {len(training_dataset)}, Validation dataset: {len(validation_dataset)}, Test dataset: {len(test_dataset)}")