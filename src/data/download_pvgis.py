import os
import time
from src.config import file_config
from data_acquisition import download_pvgis_data




os.makedirs(file_config.raw_data_dir, exist_ok=True)

for loc in file_config.locations:
    name = str(loc["name"])
    output_file = f"{file_config.raw_data_dir}/pvgis_{name.lower()}.csv"
    
    if os.path.exists(output_file):
            print(f"Skipping {output_file} — already exists")
            continue
    download_pvgis_data(float(loc["lat"]), float(loc["lon"]), 2015, 2023, output_file)

    time.sleep(2)