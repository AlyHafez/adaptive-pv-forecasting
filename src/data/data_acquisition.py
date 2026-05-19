"""download data from PVGIS and save to local disk"""
import requests  # type: ignore
import logging
import pandas as pd # type: ignore[import]
import polars as pl # type: ignore[import]
import os
from huggingface_hub import hf_hub_download # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
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


def get_london_pv_system():
    metadata_path = hf_hub_download(
        repo_id="openclimatefix/uk_pv",
        filename="metadata.csv",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
    metadata = pl.read_csv(metadata_path)
    
    london = metadata.filter(
        (pl.col("latitude_rounded").is_between(51.3, 51.7)) &
        (pl.col("longitude_rounded").is_between(-0.5, 0.3)) &
        (pl.col("kWp") <= 4.0) &
        (pl.col("end_datetime_GMT") >= "2023-12-01")
    )
    
    system = london.row(0, named=True)
    logging.info(f"Selected system {system['ss_id']} at lat={system['latitude_rounded']}, lon={system['longitude_rounded']}, kWp={system['kWp']}")
    return system["ss_id"], system["latitude_rounded"], system["longitude_rounded"]

def download_ukpv_system(ss_id: int, year: int = 2023) -> pd.DataFrame:
    df = pl.scan_parquet(
        "hf://datasets/openclimatefix/uk_pv/30_minutely",
        storage_options={"token": os.getenv("HF_TOKEN")}
    ).filter(
        (pl.col("ss_id") == ss_id) &
        (pl.col("datetime_GMT").dt.year() == year)
    ).collect().to_pandas()

    df["datetime_GMT"] = pd.to_datetime(df["datetime_GMT"])
    df = df.set_index("datetime_GMT")
    df = df["generation_Wh"].resample("1h").sum().reset_index()
    df["power_kw"] = df["generation_Wh"] / 4000.0
    
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
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,shortwave_radiation,diffuse_radiation,direct_radiation,windspeed_10m,sunshine_duration",
        "timezone": "UTC"
    }
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    logging.info(f"Downloaded {len(df)} hourly weather records")
    return df

if __name__ == "__main__":
    ss_id, lat, lon = get_london_pv_system()
    df = download_ukpv_system(ss_id, year=2023)
    print(df.head())
    print(f"Shape: {df.shape}")
    weather_df = download_openmeteo(lat, lon)
    print(weather_df.head())