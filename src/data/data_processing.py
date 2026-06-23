import pandas as pd  # type: ignore[import]
import os
import logging
from src.config import file_config
logging.basicConfig(level=logging.INFO)
def clean_data(name:str)-> pd.DataFrame:
    os.makedirs(file_config.processed_data_dir, exist_ok=True)
    
        
    output_file = f"{file_config.raw_data_dir}/pvgis_{name.lower()}.csv"
    if os.path.exists(output_file):

        df = pd.read_csv(output_file, skiprows=10, skipfooter=6, header=0, engine="python")


        df = df[df["time"].str.match(r"^\d{8}:\d{4}$", na=False)]
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
        df = df.dropna(subset=["P"])

        df["P"] = pd.to_numeric(df["P"], errors="coerce")


        df = df.drop(columns=["Int"])

        return df

            
    else:
        raise FileNotFoundError(f"Raw data file not found: {output_file}")

        
def process_load_data(file_path:str)-> pd.DataFrame:
    """Process the load data from the specified file path.
    Args:
        file_path (str): Path to the load data file.
    Returns:
        pd.DataFrame: A DataFrame containing the processed load data/
        """
    
    load_df = pd.read_parquet(file_path)
    load_df = load_df.rename(columns={'ts_utc': 'datetime'})
    load_df['datetime'] = pd.to_datetime(load_df['datetime'])
    load_df = load_df.sort_values(['id', 'datetime'])
    load_df_sorted = (
        load_df
        .set_index('datetime')
        .groupby('id')['value_Wh']
        .resample('h')
        .sum()
        .reset_index()
    )
    load_df_sorted['load_kWh'] = load_df_sorted['value_Wh'] / 1000  # Convert Wh to kWh



    load_df_sorted.drop_duplicates(subset=['id', 'datetime'], keep='first', inplace=True)

    load_df_sorted.to_parquet(file_config.processed_load_data_path, index=False)
    # Add processing logic here if needed
    return load_df_sorted

def load_data(path:str)->pd.DataFrame:
    df = pd.read_parquet(path)
    logging.info(f"Loaded data from {path}: {df.shape}")
    return df 
    
if __name__ == "__main__":
    for loc in file_config.locations:
        df = clean_data(name=str(loc["name"]))
        name = str(loc["name"])
        df.to_csv(f"{file_config.processed_data_dir}/pvgis_{name.lower()}_processed.csv")
    load_df = process_load_data(file_config.load_data_path)