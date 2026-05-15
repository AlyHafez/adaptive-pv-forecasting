import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import os
import logging
from data_processing import clean_data
from data_config import file_config
logging.basicConfig(level=logging.INFO)
def add_features(location:str, lat:float, lon:float)->pd.DataFrame:
    """
    adds time-based features, daylight indicator, and normalized power output to the processed PVGIS data for a given location.
    Args:
        location (str): Name of the location (e.g., "London").
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
    Returns:
        pd.DataFrame: DataFrame with added features 
    """
    df = clean_data(location)
    df["location"] = location
    df["lat"] = lat
    df["lon"] = lon
    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["time"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["time"].dt.month / 12)

    df["dayofyear_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365)

    df["daylight"] = (df["H_sun"] > 0).astype(int)

    df["P_norm"] = df["P"] / 1000.0

    df["time_idx"] = range(len(df))

    return df

if __name__ == "__main__":
    """
    Main function to process all locations, add features, and save the results. 
    It also combines all location data into a single CSV file for easier analysis.
    """
    os.makedirs(file_config.processed_data_dir, exist_ok=True)
    all_dfs = []
    for loc in file_config.locations:
        name = str(loc["name"])
        lat = float(loc["lat"])
        lon = float(loc["lon"])
        df = add_features(name, lat, lon)
        logging.info(f"extracted features for {name}: {df.shape}")
        all_dfs.append(df)
    df_all = pd.concat(all_dfs).reset_index(drop=True)
    df_all.to_parquet(f"{file_config.processed_data_dir}/pvgis_all.parquet", index=False)
    logging.info(f"Combined: {df_all.shape}")
    logging.info(df_all["location"].value_counts())
    