import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import os
import logging
from src.data.data_acquisition import download_openmeteo, download_ukpv_system,  get_pv_system, get_pv_systems_multiple
from src.data.data_processing import clean_data
from src.config import file_config 
logging.basicConfig(level=logging.INFO)
def add_features(location: str, lat: float, lon: float, df: pd.DataFrame | None =  None) -> pd.DataFrame:
    
    if df is None:
        df = clean_data(location)  # original PVGIS flow
        df["location"] = location.lower()
        df["series_id"] = df["location"]
    else: 
        df["location"] = location.lower()
        df["series_id"] =  df["location"] + df["ss_id"].astype(str) # for UK_PV systems, create unique series_id by combining location and ss_id
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


def prepare_ukpv_as_pvgis(
    pv_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    ss_id: int,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """Prepare UK_PV + Open-Meteo data to match PVGIS format so it can be 
    passed directly to add_features().
    Args:
        pv_df: hourly UK_PV generation data
        weather_df: hourly Open-Meteo weather data
        ss_id: Sheffield Solar ID
        lat, lon: coordinates
    Returns:
        pd.DataFrame: DataFrame matching PVGIS format ready for add_features()
    """
    # align timestamps
    pv_df["datetime_GMT"] = pd.to_datetime(pv_df["datetime_GMT"]).dt.tz_localize(None)
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    # merge
    df = pd.merge(
        pv_df.rename(columns={"datetime_GMT": "time"}),
        weather_df,
        on="time",
        how="inner"
    ) 

    # rename to match PVGIS column names exactly
    df = df.rename(columns={
        "time":              "time",
        "temperature_2m":    "T2m",
        "windspeed_10m":     "WS10m",
        "direct_radiation":  "Gb(i)",
        "diffuse_radiation": "Gd(i)",
        "sunshine_duration": "H_sun",
    })

    # compute Gr(i)
    df["Gr(i)"] = (df["shortwave_radiation"] - df["Gb(i)"] - df["Gd(i)"]).clip(lower=0)

    # P in watts to match PVGIS
    df["P"] = df["power_kw"] * 1000

    # keep only PVGIS columns
    cols = ["time", "P", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m"]
    return df[cols].dropna().reset_index(drop=True)



if __name__ == "__main__":
    # prepare pvgis data with features for TFT training
    os.makedirs(file_config.processed_data_dir, exist_ok=True)
    all_dfs = []
    for loc in file_config.locations:
        name = str(loc["name"])
        lat = float(loc["lat"])
        lon = float(loc["lon"])
        df = add_features(name, lat, lon)
        logging.info(f"extracted features for {name}: {df.shape}")
        all_dfs.append(df)
    df_pvgis_all = pd.concat(all_dfs).reset_index(drop=True)
    df_pvgis_all.to_parquet(f"{file_config.processed_data_dir}/pvgis_all.parquet", index=False) 
    logging.info(f"Combined: {df_pvgis_all.shape}")
    logging.info(f"\n{df_pvgis_all.head()}")
    logging.info(df_pvgis_all["location"].value_counts())

    # Example usage: prepare UK_PV data for one system and save, then prepare for multiple systems and save, then also prepare original PVGIS data with features for TFT training.
    ss_id, lat, lon, kwp = get_pv_system()
    pv_df = download_ukpv_system(ss_id, kwp, year=2023)
    weather_df = download_openmeteo(lat, lon)
    pvgis_format_df = prepare_ukpv_as_pvgis(pv_df, weather_df, ss_id, lat, lon)
    pvgis_format_df["ss_id"] = ss_id  # add ss_id for reference
    df = add_features(f"London", lat, lon, df=pvgis_format_df)
    df.to_parquet(f"{file_config.processed_data_dir}/ukpv_london_test_household.parquet", index=False)
    logging.info(f"Saved: {df.shape}")
    logging.info(f"\n{df.head()}")

    systems = get_pv_systems_multiple(n=10, exclude_ss_id=2746)
    df_all: list[pd.DataFrame] = []
    for system in systems.iter_rows(named=True):
        ss_id = system["ss_id"]
        lat = system["latitude_rounded"]
        lon = system["longitude_rounded"]
        kwp = system["kWp"]
        tft_df = download_ukpv_system(ss_id, kwp, year=2023)
        pvgis_format_df = prepare_ukpv_as_pvgis(tft_df, weather_df, ss_id, lat, lon)
        pvgis_format_df["ss_id"] = ss_id  # add ss_id for reference
        tft_df = add_features("London", lat, lon, df=pvgis_format_df)
        
        df_all.append(tft_df)
    df_combined = pd.concat(df_all, ignore_index=True)
    logging.info(df_combined.head())
    print(f"Shape: {df_combined.shape}")
   
    df_combined.to_parquet(f"{file_config.processed_data_dir}/ukpv_london_tft.parquet", index=False)
    logging.info(f"Saved: {df_combined.shape}")
    logging.info(f"\n{df_combined.head()}")

