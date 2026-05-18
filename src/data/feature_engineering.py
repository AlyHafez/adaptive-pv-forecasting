import pandas as pd # type: ignore[import]
import numpy as np # type: ignore[import]
import os
import logging
from src.data.data_acquisition import download_openmeteo, download_ukpv_system,  get_london_pv_system
from src.data.data_processing import clean_data
from src.config import file_config 
logging.basicConfig(level=logging.INFO)
def add_features(location: str, lat: float, lon: float, df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = clean_data(location)  # original PVGIS flow
    
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
    ss_id, lat, lon = get_london_pv_system()
    pv_df = download_ukpv_system(ss_id, year=2023)
    weather_df = download_openmeteo(lat, lon)
    
    pvgis_format_df = prepare_ukpv_as_pvgis(pv_df, weather_df, ss_id, lat, lon)
    df = add_features(f"London", lat, lon, df=pvgis_format_df)
    
    os.makedirs(file_config.processed_data_dir, exist_ok=True)
    df.to_parquet(f"{file_config.processed_data_dir}/ukpv_london_tft.parquet", index=False)
    logging.info(f"Saved: {df.shape}")
    logging.info(f"\n{df.head()}")