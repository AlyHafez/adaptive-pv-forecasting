import pandas as pd
import numpy as np
import os
from data_config import file_config
def add_features(location:str)->pd.DataFrame:
    df = pd.read_csv(f"{file_config.processed_data_dir}/pvgis_{location.lower()}_processed.csv", parse_dates=["time"])
    df["location"] = location

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
    os.makedirs(file_config.processed_data_dir, exist_ok=True)
    all_dfs = []
    for loc in file_config.locations:
        name = str(loc["name"])
        df = add_features(name)
        df.to_csv(f"{file_config.processed_data_dir}/pvgis_{name.lower()}_features.csv", index=False)
        print(f"Saved {name}: {df.shape}")
        all_dfs.append(df)
    df_all = pd.concat(all_dfs).reset_index(drop=True)
    df_all.to_csv(f"{file_config.processed_data_dir}/pvgis_all.csv", index=False)
    print(f"Combined: {df_all.shape}")
    print(df_all["location"].value_counts())
    print(df_all.head())