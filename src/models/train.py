from lightning.pytorch import Trainer # type: ignore[import]
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint # type: ignore[import]
from lightning.pytorch.loggers import WandbLogger   # type: ignore[import]
import logging # type: ignore[import]
import wandb# type: ignore[import]
from dotenv import load_dotenv# type: ignore[import]
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet# type: ignore[import]
import os# type: ignore[import]
import pandas as pd# type: ignore[import]
from src.models.tft import create_tft# type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader
from src.config import file_config, model_config
from src.utils.utils import wandb_login

logging.basicConfig(level=logging.INFO)


def train(epochs: int, accelerator: str = "auto", gradient_clip_val: float = 0.9):
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, _ = split_data(data)
    
    training_dataset = create_dataset(train_data, model_config.max_encoder_length, model_config.max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    
    train_dataloader = dataloader(training_dataset, batch_size=model_config.train_batch_size, train=True)
    val_dataloader = dataloader(validation_dataset, batch_size=model_config.val_batch_size, train=False)
    
    tft = create_tft(training_dataset)
    logging.info(f"Number of parameters in model: {tft.size()/1e6:.2f} million")
    
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, mode="min")
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{file_config.models_dir}/tft",
        filename="tft-best-model",
        save_top_k=1,
        mode="min"
    )
    wandb_logger = WandbLogger(project="pv-forecasting", name="tft-training")
    
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop, checkpoint],
        logger=wandb_logger,
    )
    
    trainer.fit(tft, train_dataloader, val_dataloader)
    return trainer, tft


if __name__ == "__main__":
    wandb_login()
    os.makedirs(f"{file_config.models_dir}/tft", exist_ok=True)
    train(epochs=50, accelerator="auto", gradient_clip_val=model_config.grad_clip_val)