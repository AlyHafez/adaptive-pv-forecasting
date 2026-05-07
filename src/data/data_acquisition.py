"""download data from PVGIS and save to local disk"""
import requests  # type: ignore

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
        print(f"Failed to download data: {request.status_code} — {request.text}")


