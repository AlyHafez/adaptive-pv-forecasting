import pandas as pd  # type: ignore[import]
import os
from data_config import file_config
os.chdir("/Users/alyshoukry/Desktop/MSc/Dissertation/adaptive-pv-forecasting")
for loc in file_config.locations:
    name = str(loc["name"])
    output_file = f"{file_config.raw_data_dir}/pvgis_{name.lower()}.csv"
    if os.path.exists(output_file):
        print(f"Loaded {output_file}")
        df = pd.read_csv(output_file, skiprows=10, skipfooter=6, header=0, engine="python")
        print(f"{name} data shape: {df.shape}")

        df = df[df["time"].str.match(r"^\d{8}:\d{4}$", na=False)]
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
        df = df.dropna(subset=["P"])

        df["P"] = pd.to_numeric(df["P"], errors="coerce")

        print(df.head())
        df = df.drop(columns=["Int"])

        print(df.describe())

        print(df.isna().sum())
        print(df.duplicated(subset=["time"]).sum())
        df = df.set_index("time").sort_index()
        expected = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        missing = expected.difference(df.index)
        print(f"Missing timestamps: {len(missing)}")
        print((df[["P", "Gb(i)", "Gd(i)", "Gr(i)"]] < 0).sum())
        df.to_csv(f"{file_config.processed_data_dir}/pvgis_{name.lower()}_processed.csv")
    else:
        print(f"File not found: {output_file}")