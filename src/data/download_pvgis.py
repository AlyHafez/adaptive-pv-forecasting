import os
import time
from data_acquisition import download_pvgis_data


locations: list[dict[str, str | float]]= [
    {"lat": 51.5,  "lon": -0.1,  "name": "London"},
    {"lat": 48.8,  "lon": 2.3,   "name": "Paris"},
    {"lat": 52.5,  "lon": 13.4,  "name": "Berlin"},
    {"lat": 40.4,  "lon": -3.7,  "name": "Madrid"},
    {"lat": 41.9,  "lon": 12.5,  "name": "Rome"},
    {"lat": 52.3,  "lon": 4.9,   "name": "Amsterdam"},
    {"lat": 38.7,  "lon": -9.1,  "name": "Lisbon"},
    {"lat": 52.2,  "lon": 21.0,  "name": "Warsaw"},
    {"lat": 59.3,  "lon": 18.1,  "name": "Stockholm"},
    {"lat": 45.7,  "lon": 4.8,   "name": "Lyon"},
]

os.makedirs("data/raw", exist_ok=True)

for loc in locations:
    name = str(loc["name"])
    output_file = f"data/raw/pvgis_{name.lower()}.csv"
    
    if os.path.exists(output_file):
            print(f"Skipping {output_file} — already exists")
            continue
    download_pvgis_data(float(loc["lat"]), float(loc["lon"]), 2015, 2023, output_file)

    time.sleep(2)