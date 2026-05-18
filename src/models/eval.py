import pandas as pd # type: ignore[import]
import logging # type: ignore[import]

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet # type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader
from src.config import file_config, model_config
from src.utils.utils import wandb_login

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
        test_dataloader, mode="quantiles",
        return_x=True, return_y=True, return_index=True,
        show_progress_bar=True,
        trainer_kwargs=dict(accelerator="auto"),
    )
    return predictions


if __name__ == "__main__":
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, test_data = split_data(data)
    training_dataset = create_dataset(train_data, model_config.max_encoder_length, model_config.max_prediction_length)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
    test_dataloader = dataloader(test_dataset, batch_size=model_config.test_batch_size, train=False)
    
    checkpoint_path = f"{file_config.models_dir}/tft/tft-best-model.ckpt"  
    best_model = load_model(checkpoint_path)
    predictions = evaluate_tft(best_model, test_dataloader)
    pd.DataFrame({
    "actual": predictions.y[0].numpy().flatten(),
    "median": predictions.output[:, :, 1].numpy().flatten(),
    "lower": predictions.output[:, :, 0].numpy().flatten(),
    "upper": predictions.output[:, :, 2].numpy().flatten(),
}).to_csv(f"{file_config.results_dir}/predictions_tft.csv", index=False)