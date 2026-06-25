import pandas as pd # type: ignore[import]
import logging # type: ignore[import]
import wandb # type: ignore
import numpy as np # type: ignore[import]
import sys

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet # type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader
from src.config import file_config, tft_config
from src.utils.utils import wandb_login


logging.basicConfig(level=logging.INFO)
def load_model(checkpoint_path: str) -> TemporalFusionTransformer:
    """Load a trained Temporal Fusion Transformer model from a checkpoint.
    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
    Returns:
        TemporalFusionTransformer: The loaded model ready for evaluation."""
    return TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
def evaluate_tft(best_model: TemporalFusionTransformer, test_dataloader: TimeSeriesDataSet.to_dataloader):
    """Evaluate a trained TFT model on the test dataset.
    Args:
        best_model (TemporalFusionTransformer): The trained model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
    Returns:
        tuple: A tuple containing the predictions, targets, and quantile predictions.
    """
    

    predictions = best_model.predict(
        test_dataloader, mode="raw",
        return_x=True, return_y=True, return_index=True,
        trainer_kwargs=dict(accelerator="auto",devices=1),
    )
    interpretation = best_model.interpret_output(predictions.output, reduction="mean")
    print(interpretation.keys())
    print(interpretation["encoder_variables"])
    fig = best_model.plot_interpretation(interpretation)
    wandb.log({"interpretation": fig}) #type:ignore

    return predictions

def simulate_drift(df:pd.DataFrame, drift_magnitude: float, drift_start:pd.Timestamp)-> pd.DataFrame:
    """Simulate a drift in the target variable by adding a time-varying bias.
    Args:
        df (pd.DataFrame): The input dataframe containing the original data.
        drift_magnitude (float): The maximum magnitude of the drift to simulate.
        drift_start (datetime): The start time of the drift.
    Returns:
        pd.DataFrame: A new dataframe with the simulated drift added to the target variable."""
    df_drifted = df.copy()
    times = df["time"]
    drift_idx = (times >= drift_start).argmax()  # index where drift starts
    n = len(df) - drift_idx

    drift = np.ones(len(df))
    drift[drift_idx:] = np.linspace(1.0, 1.0 - drift_magnitude, n)
    df_drifted["P_norm"] = df["P_norm"]*drift
    logging.info(f"Drift starts at {drift_start} (index {drift_idx})")
    logging.info(f"Final drift factor: {drift[-1]:.3f} ({drift_magnitude*100:.0f}% reduction)")
    return df_drifted
def simulate_partial_shading(df:pd.DataFrame, shading_magnitude: float, shading_start:pd.Timestamp)-> pd.DataFrame:
    """Simulate partial shading by adding a  bias to the target variable.
    Args:
        df (pd.DataFrame): The input dataframe containing the original data.
        shading_magnitude (float): The maximum magnitude of the shading to simulate.
        shading_start (datetime): The start time of the shading.
    Returns:
        pd.DataFrame: A new dataframe with the simulated shading added to the target variable."""
    df_shaded = df.copy()
    times = df["time"]
    shading_idx = (times >= shading_start)  # index where shading starts
    

    df_shaded.loc[shading_idx, "P_norm"] = df.loc[shading_idx, "P_norm"] * (1-shading_magnitude)
    logging.info(f"Shading starts at {shading_start}")
    logging.info(f"Final shading factor: ({shading_magnitude*100:.0f}% reduction)")
    return df_shaded

if __name__ == "__main__":
    wandb_login()
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "pvgis"
    
    # set checkpoint and output file based on mode
    if mode == "pvgis":
        checkpoint_path = file_config.tft_checkpoint_path
        output_file = "predictions_pvgis.csv"
        run_name = "eval-pvgis"
    elif mode == "ukpv_finetuned":
        checkpoint_path = file_config.fine_tuned_path
        output_file = "predictions_ukpv_finetuned.csv"
        output_file2 = "predictions_ukpv2_finetuned.csv"
        run_name = "eval-ukpv-finetuned"
    elif mode == "ukpv_pretrained":
        checkpoint_path = file_config.tft_checkpoint_path
        output_file = "predictions_ukpv_pretrained.csv"
        output_file2 = "predictions_ukpv2_pretrained.csv"
        run_name = "eval-ukpv-pretrained"
    elif mode == "ukpv_finetuned_drifted":
        checkpoint_path =file_config.fine_tuned_path
        output_file = "predictions_ukpv_finetuned_drifted.csv"
        output_file2 = "predictions_ukpv2_finetuned_drifted.csv"
        run_name = "eval-ukpv-finetuned-drifted"
    elif mode == "ukpv_pretrained_drifted":
        checkpoint_path = file_config.tft_checkpoint_path
        output_file = "predictions_ukpv_pretrained_drifted.csv"
        output_file2 = "predictions_ukpv2_pretrained_drifted.csv"
        run_name = "eval-ukpv-pretrained-drifted"
    elif mode == "ukpv_finetuned_shaded":
        checkpoint_path = file_config.fine_tuned_path
        output_file = "predictions_ukpv_finetuned_shaded.csv"
        output_file2 = "predictions_ukpv2_finetuned_shaded.csv"
        run_name = "eval-ukpv-finetuned-shaded"
    elif mode == "ukpv_pretrained_shaded":
        checkpoint_path = file_config.tft_checkpoint_path
        output_file = "predictions_ukpv_pretrained_shaded.csv"
        output_file2 = "predictions_ukpv2_pretrained_shaded.csv"
        run_name = "eval-ukpv-pretrained-shaded"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    wandb.init(project="pv-forecasting", name=run_name) #type: ignore 
    
    data = pd.read_parquet(file_config.data_path)
    
    train_data, val_data, _ = split_data(data)
    training_dataset = create_dataset(train_data, tft_config.max_encoder_length, tft_config.max_prediction_length)
    best_model = load_model(checkpoint_path)

    if mode == "pvgis":
        _, _, test_data = split_data(data)
        test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True, predict=True)
        test_dataloader = dataloader(test_dataset, batch_size=tft_config.test_batch_size, train=False)
        predictions = evaluate_tft(best_model, test_dataloader)
        quantiles = predictions.output.prediction
        df_out = pd.DataFrame({
            "time": data["time"].values[168+23:168+23+len(quantiles)],
            "actual": predictions.y[0].cpu().numpy()[:, 23],
            "median": quantiles[:, 23, 1].cpu().numpy(),
            "lower": quantiles[:, 23, 0].cpu().numpy(),
            "upper": quantiles[:, 23, 2].cpu().numpy(),
        })
        df_out["median"] = df_out["median"].clip(lower=0)
        df_out["lower"] = df_out["lower"].clip(lower=0)
        df_out["upper"] = df_out["upper"].clip(lower=0)
        df_out.to_csv(f"{file_config.results_dir}/{output_file}", index=False)
        logging.info(f"PVGIS predictions saved to {output_file}")

    else:  # ukpv_finetuned or ukpv_pretrained
        ukpv_df = pd.read_parquet(file_config.test_set)
        ukpv2_df = pd.read_parquet(file_config.household2_test_set)
        if mode in ["ukpv_finetuned_drifted", "ukpv_pretrained_drifted"]:
            ukpv_df = simulate_drift(ukpv_df, drift_magnitude=0.25, drift_start=pd.Timestamp("2023-06-01 00:00:00"))
            ukpv_df.to_parquet(f"{file_config.test_set_drifted}", index=False)
            ukpv2_df = simulate_drift(ukpv2_df,drift_magnitude=0.25, drift_start=pd.Timestamp("2023-06-01 00:00:00"))
            ukpv2_df.to_parquet(f"{file_config.household2_test_set_drifted}", index=False)
            
        if mode in ["ukpv_finetuned_shaded", "ukpv_pretrained_shaded"]:
            ukpv_df = simulate_partial_shading(ukpv_df, shading_magnitude=0.4153, shading_start=pd.Timestamp("2023-06-01 00:00:00"))
            ukpv_df.to_parquet(f"{file_config.test_set_shaded}", index=False)
            ukpv2_df = simulate_partial_shading(ukpv2_df, shading_magnitude=0.4153, shading_start=pd.Timestamp("2023-06-01 00:00:00"))
            ukpv2_df.to_parquet(f"{file_config.household2_test_set_shaded}", index=False)

        ukpv_df["series_id"] = ukpv_df["location"]
        ukpv_df = ukpv_df.reset_index(drop=True)
        ukpv_df["time_idx"] = range(len(ukpv_df))
        logging.info("predictions for household 1 kwp:4.0")
        ukpv_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, ukpv_df,
            predict=False, stop_randomization=True, allow_missing_timesteps=True
        )
        ukpv_dataloader = dataloader(ukpv_dataset, batch_size=tft_config.test_batch_size, train=False)
        predictions = evaluate_tft(best_model, ukpv_dataloader)
        quantiles = predictions.output.prediction
        df_out = pd.DataFrame({
            "time": ukpv_df["time"].values[168+23:168+23+len(quantiles)],
            "actual": predictions.y[0].cpu().numpy()[:, 23],
            "median": quantiles[:, 23, 1].cpu().numpy(),
            "lower": quantiles[:, 23, 0].cpu().numpy(),
            "upper": quantiles[:, 23, 2].cpu().numpy(),
        })
        df_out["median"] = df_out["median"].clip(lower=0)
        df_out["lower"] = df_out["lower"].clip(lower=0)
        df_out["upper"] = df_out["upper"].clip(lower=0)
        df_out.to_csv(f"{file_config.results_dir}/{output_file}", index=False)
        logging.info(f"UK_PV household 1 predictions saved to {output_file}")

        
        ukpv2_df["series_id"] = ukpv2_df["location"]
        ukpv2_df = ukpv_df.reset_index(drop=True)
        ukpv2_df["time_idx"] = range(len(ukpv2_df))
        logging.info("predictions for household 1 kwp:2.3")
        ukpv2_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, ukpv2_df,
            predict=False, stop_randomization=True, allow_missing_timesteps=True
        )
        ukpv2_dataloader = dataloader(ukpv2_dataset, batch_size=tft_config.test_batch_size, train=False)
        predictions = evaluate_tft(best_model, ukpv2_dataloader)
        quantiles = predictions.output.prediction
        df_out = pd.DataFrame({
            "time": ukpv2_df["time"].values[168+23:168+23+len(quantiles)],
            "actual": predictions.y[0].cpu().numpy()[:, 23],
            "median": quantiles[:, 23, 1].cpu().numpy(),
            "lower": quantiles[:, 23, 0].cpu().numpy(),
            "upper": quantiles[:, 23, 2].cpu().numpy(),
        })
        df_out["median"] = df_out["median"].clip(lower=0)
        df_out["lower"] = df_out["lower"].clip(lower=0)
        df_out["upper"] = df_out["upper"].clip(lower=0)
        df_out.to_csv(f"{file_config.results_dir}/{output_file2}", index=False)
        logging.info(f"UK_PV household 2 predictions saved to {output_file2}")

    wandb.finish()  # type: ignore