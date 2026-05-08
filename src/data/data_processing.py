import pandas as pd  # type: ignore[import]
import os
from data_config import file_config
os.chdir("/Users/alyshoukry/Desktop/MSc/Dissertation/adaptive-pv-forecasting")
def clean_data():
    for loc in file_config.locations:
        name = str(loc["name"])
        output_file = f"{file_config.raw_data_dir}/pvgis_{name.lower()}.csv"
        if os.path.exists(output_file):

            df = pd.read_csv(output_file, skiprows=10, skipfooter=6, header=0, engine="python")


            df = df[df["time"].str.match(r"^\d{8}:\d{4}$", na=False)]
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
            df = df.dropna(subset=["P"])

            df["P"] = pd.to_numeric(df["P"], errors="coerce")


            df = df.drop(columns=["Int"])



            df = df.set_index("time").sort_index()



            df.to_csv(f"{file_config.processed_data_dir}/pvgis_{name.lower()}_processed.csv")
        else:
            print(f"File not found: {output_file}")
    
if __name__ == "__main__":
    clean_data()