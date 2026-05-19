from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet # type: ignore[import]
from src.utils.utils import set_seed # type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader  # type: ignore[import]
from src.config import file_config, tft_config # type: ignore[import]
import pandas as pd # type: ignore[import]
import logging
from pytorch_forecasting.metrics import QuantileLoss # type: ignore[import]
from lightning.pytorch import Trainer # type: ignore[import]


logging.basicConfig(level=logging.INFO)

def create_network():
    set_seed(tft_config.seed)
    return Trainer(
        accelerator="auto",
        gradient_clip_val=tft_config.grad_clip_val,
        logger=False,
    )
def create_tft(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Create a Temporal Fusion Transformer model with specified hyperparameters.
    Args:
        training_dataset (TimeSeriesDataSet): The training dataset to determine input sizes.
    Returns:
        TemporalFusionTransformer: An instance of the Temporal Fusion Transformer model ready for training."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=tft_config.lr,
        hidden_size=tft_config.hidden_size,
        attention_head_size=tft_config.attention_heads,
        dropout=tft_config.dropout,
        hidden_continuous_size=tft_config.hidden_continuous_size,
        output_size=3,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        optimizer="ranger",                  
        reduce_on_plateau_patience=4,        
        log_interval=10,                     
    )

if __name__ == "__main__":
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, test_data = split_data(data)
    logging.info(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    max_encoder_length = tft_config.max_encoder_length# use the past week of data
    max_prediction_length = tft_config.max_prediction_length # predict the next 24 hours
    training_dataset = create_dataset(train_data, max_encoder_length, max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_data, stop_randomization=True)
    
    train_dataloader = dataloader(training_dataset, batch_size=tft_config.train_batch_size, train=True, num_workers=tft_config.num_workers)
    val_dataloader = dataloader(validation_dataset, batch_size=tft_config.val_batch_size, train=False, num_workers=tft_config.num_workers)
    test_dataloader = dataloader(test_dataset, batch_size=tft_config.test_batch_size, train=False, num_workers=tft_config.num_workers)
    tft = create_tft(training_dataset)
    trainer = create_network()
    logging.info(f"number of parameters in model: {tft.size()/1e6:.2f} million")