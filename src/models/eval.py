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
    print(interpretation.keys())
    print(interpretation["encoder_variables"])
    fig = best_model.plot_interpretation(interpretation)
    wandb.log({"interpretation": fig}) #type:ignore

    return predictions



    
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
        run_name = "eval-ukpv-finetuned"
    elif mode == "ukpv_pretrained":
        checkpoint_path = file_config.tft_checkpoint_path
        output_file = "predictions_ukpv_pretrained.csv"
        run_name = "eval-ukpv-pretrained"
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
        pd.DataFrame({
            "actual": predictions.y[0].cpu().numpy()[:, 0],
            "median": quantiles[:, 0, 1].cpu().numpy(),
            "lower": quantiles[:, 0, 0].cpu().numpy(),
            "upper": quantiles[:, 0, 2].cpu().numpy(),
        }).to_csv(f"{file_config.results_dir}/{output_file}", index=False)
        logging.info(f"PVGIS predictions saved to {output_file}")

    else:  # ukpv_finetuned or ukpv_pretrained
        ukpv_df = pd.read_parquet(file_config.test_set)
        ukpv_df["series_id"] = ukpv_df["location"]
        ukpv_df = ukpv_df.reset_index(drop=True)
        ukpv_df["time_idx"] = range(len(ukpv_df))
        ukpv_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, ukpv_df,
            predict=False, stop_randomization=True, allow_missing_timesteps=True
        )
        ukpv_dataloader = dataloader(ukpv_dataset, batch_size=tft_config.test_batch_size, train=False)
        predictions = evaluate_tft(best_model, ukpv_dataloader)
        quantiles = predictions.output.prediction
        pd.DataFrame({
            "actual": predictions.y[0].cpu().numpy()[:, 0],
            "median": quantiles[:, 0, 1].cpu().numpy(),
            "lower": quantiles[:, 0, 0].cpu().numpy(),
            "upper": quantiles[:, 0, 2].cpu().numpy(),
        }).to_csv(f"{file_config.results_dir}/{output_file}", index=False)
        logging.info(f"UK_PV predictions saved to {output_file}")

    wandb.finish()  # type: ignore