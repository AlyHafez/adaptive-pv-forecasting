import pandas as pd # type: ignore[import]
import logging # type: ignore[import]
import wandb # type: ignore
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
    fig = best_model.plot_interpretation(interpretation)
    wandb.log({"interpretation": fig}) #type:ignore

    return predictions


if __name__ == "__main__":
    
    wandb_login()
    wandb.init(project="pv-forecasting", name="feature-interpretation")  # type: ignore



    # check if running on PVGIS test set or UK_PV
    mode = sys.argv[1] if len(sys.argv) > 1 else "pvgis"
    
    # load training dataset reference (needed for both modes)
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, _ = split_data(data)
    training_dataset = create_dataset(train_data, tft_config.max_encoder_length, tft_config.max_prediction_length)
    
    checkpoint_path = f"{file_config.fine_tuned_path}"
    best_model = load_model(checkpoint_path)
    
    if mode == "pvgis":
        # original PVGIS test set evaluation
        _, _, test_data = split_data(data)
        test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
        test_dataloader = dataloader(test_dataset, batch_size=tft_config.test_batch_size, train=False)
        predictions = evaluate_tft(best_model, test_dataloader)
        quantiles = predictions.output.prediction
        pd.DataFrame({
            "actual": predictions.y[0].cpu().numpy().flatten(),
            "median": quantiles[:, :, 1].cpu().numpy().flatten(),
            "lower": quantiles[:, :, 0].cpu().numpy().flatten(),
            "upper": quantiles[:, :, 2].cpu().numpy().flatten(),
        }).to_csv(f"{file_config.results_dir}/predictions_tft.csv", index=False)
        logging.info("PVGIS predictions saved")

    elif mode == "ukpv":
        # UK_PV 37-day inference
        ukpv_df = pd.read_parquet(f"{file_config.test_set}")
        ukpv_df["series_id"] = ukpv_df["location"]
        # last 37 days
        last_date = ukpv_df["time"].max()
        ukpv_37 = ukpv_df.reset_index(drop=True)
        ukpv_37["time_idx"] = range(len(ukpv_37))
        
        # create dataset from training reference
        ukpv_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            ukpv_37,
            predict=False,
            stop_randomization=True,
            allow_missing_timesteps=True
        )
        ukpv_dataloader = dataloader(ukpv_dataset, batch_size=tft_config.test_batch_size, train=False)
        predictions = evaluate_tft(best_model, ukpv_dataloader)
        
        # save train predictions (30 days) for MLP
        pd.DataFrame({
            "actual": predictions.y[0].cpu().numpy().flatten(),
            "median": predictions.output[:, :, 1].cpu().numpy().flatten(),
            "lower": predictions.output[:, :, 0].cpu().numpy().flatten(),
            "upper": predictions.output[:, :, 2].cpu().numpy().flatten(),
        }).to_csv(f"{file_config.results_dir}/predictions_ukpv.csv", index=False)
        logging.info("UK_PV predictions saved")
        wandb.finish()# type: ignore
