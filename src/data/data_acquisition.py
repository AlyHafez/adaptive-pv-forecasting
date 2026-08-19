"""download data from PVGIS and save to local disk"""
import requests  # type: ignore
import logging
import pandas as pd # type: ignore[import]
import polars as pl # type: ignore[import]
import os
from huggingface_hub import hf_hub_download # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
from src.config import file_config
load_dotenv()
logging.basicConfig(level=logging.INFO)
def download_pvgis_data(latitude:float, longitude:float, start_year:int,
                         end_year:int, output_file:str)->None:
    """Download PVGIS data for a given location and time period and save to a local file.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_year (int): Start year for the data.
        end_year (int): End year for the data.
        output_file (str): Path to the output file where data will be saved.
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params: dict[str, str | int | float] = {
        "lat": latitude,
        "lon": longitude,
        "startyear": start_year,
        "endyear": end_year,
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": 14,
        "angle": 35,
        "components": 1,
        "outputformat": "csv"
    }
    request = requests.get(url, params=params, timeout=30)
    if request.status_code == 200:
        with open(output_file, "w") as f:
            f.write(request.text)
        print(f"Saved: {output_file}")
    else:
        logging.error(f"Failed to download data for lat={latitude}, lon={longitude}. Status code: {request.status_code}")

def get_pv_systems_multiple(n: int = 10, exclude_ss_id: int = 2746,                             
                            lat_min: float = 51.3, lat_max: float = 51.7,
                             lon_min: float = -0.5, lon_max: float = 0.3):
    """Get multiple London residential PV systems for transfer learning.
    Args:
        n: number of systems to return
        exclude_ss_id: exclude the test household
    Returns:
        pl.DataFrame: metadata for n systems
    """
    metadata_path = hf_hub_download(
        repo_id="openclimatefix/uk_pv",
        filename="metadata.csv",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    metadata = pl.read_csv(metadata_path)
    
    london = metadata.filter(
        (pl.col("latitude_rounded").is_between(lat_min, lat_max)) &
        (pl.col("longitude_rounded").is_between(lon_min, lon_max)) &
        (pl.col("kWp") <= 4.0) &
        (pl.col("end_datetime_GMT") >= "2023-12-01") &
        (pl.col("ss_id") != exclude_ss_id)  # exclude test household
    )
    
    systems = london.head(n)
    logging.info(f"Found {len(systems)} for coordinate range({lat_min}, {lon_min}) to ({lat_max}, {lon_max}) systems for transfer learning")
    return systems

def get_pv_system( lat_min: float = 51.3, lat_max: float = 51.7,
                  lon_min: float = -0.5, lon_max: float = 0.3, max_kwp:float = 4.0,
                  exclude_ss_id: list[int] | None = None):
    metadata_path = hf_hub_download(
        repo_id="openclimatefix/uk_pv",
        filename="metadata.csv",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    metadata = pl.read_csv(metadata_path)

    london = metadata.filter(
        (pl.col("latitude_rounded").is_between(lat_min, lat_max)) &
        (pl.col("longitude_rounded").is_between(lon_min, lon_max)) &
        (pl.col("kWp") <= max_kwp) &
        (pl.col("end_datetime_GMT") >= "2023-12-01")
    )
    if exclude_ss_id:
        london = london.filter(~pl.col("ss_id").is_in(exclude_ss_id))

    system = london.row(0, named=True)
    logging.info(f"Selected system {system['ss_id']} at lat={system['latitude_rounded']}, lon={system['longitude_rounded']}, kWp={system['kWp']}")
    return system["ss_id"], system["latitude_rounded"], system["longitude_rounded"], system["kWp"]

def download_ukpv_system(ss_id: int,kWp: float, year: int = 2023) -> pd.DataFrame:
    df = pl.scan_parquet(
        "hf://datasets/openclimatefix/uk_pv/30_minutely",
        storage_options={"token": os.getenv("HF_TOKEN")}
    ).filter(
        (pl.col("ss_id") == ss_id) &
        (pl.col("datetime_GMT").dt.year() == year)
    ).collect().to_pandas()
    df["ssid"] = df["ss_id"].astype(str)  # ensure ss_id is string for merging
    df["datetime_GMT"] = pd.to_datetime(df["datetime_GMT"])
    df = df.set_index("datetime_GMT")
    df = df["generation_Wh"].resample("1h").sum().reset_index()
    df["ss_id"] = ss_id  # use the parameter directly, no need to read from column
    df["power_kw"] = df["generation_Wh"] / (kWp * 1000)
    
    logging.info(f"Downloaded {len(df)} hourly records for system {ss_id}")
    return df

def download_openmeteo(lat: float, lon: float, start: str = "2023-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """Download historical weather data from Open-Meteo for a given location.
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start (str): Start date in YYYY-MM-DD format.
        end (str): End date in YYYY-MM-DD format.
    Returns:
        pd.DataFrame: Hourly weather data."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,shortwave_radiation,diffuse_radiation,direct_radiation,windspeed_10m,sunshine_duration",
        "timezone": "UTC",
        "models": "ukmo_seamless"
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    logging.info(f"Downloaded {len(df)} hourly weather records")
    return df

if __name__ == "__main__":
    systems = get_pv_systems_multiple(n=10, exclude_ss_id=2746)
    ss_id, lat, lon, kwp = get_pv_system(max_kwp=4.0)
    logging.info(f"Selected system {ss_id} at lat={lat}, lon={lon}, kWp={kwp}")
    df_test = download_ukpv_system(ss_id, kwp, year=2023)

    df_all: list[pd.DataFrame] = []
    for system in systems.iter_rows(named=True):
        ss_id = system["ss_id"]
        lat = system["latitude_rounded"]
        lon = system["longitude_rounded"]
        kwp = system["kWp"]
        df = download_ukpv_system(ss_id, kwp, year=2023)
        df_all.append(df)
    
    print(df_test.head())
    print(f"Shape: {df_test.shape}")
    weather_df = download_openmeteo(lat, lon)
    weather_df.to_parquet(f"{file_config.raw_data_dir}/ukpv__weather{lat}_{lon}.parquet", index=False)
    print(weather_df.head())

    