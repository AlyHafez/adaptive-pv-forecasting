from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet # type: ignore[import]
from lightning.pytorch import Trainer # type: ignore[import]
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint # type: ignore[import]
from lightning.pytorch.loggers import WandbLogger   # type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader
from src.config import file_config, tft_config
from src.utils.utils import wandb_login
from src.data.data_processing import load_data
from src.models.eval import load_model
import pandas as pd # type: ignore[import]
import logging # type: ignore[import]
import wandb # type: ignore
import os


def transfer_learning(data: pd.DataFrame, checkpoint_path: str) -> TemporalFusionTransformer:
    """Perform transfer learning by loading a pre-trained TFT model and fine-tuning it on new data.
    Args:
        data (pd.DataFrame): The new dataset to fine-tune the model on.
        checkpoint_path (str): Path to the pre-trained model checkpoint.
    Returns:
        TemporalFusionTransformer: The fine-tuned model ready for evaluation."""
    
    train_data, val_data, _ = split_data(data)
    training_dataset = create_dataset(train_data, tft_config.max_encoder_length, tft_config.max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    
    train_dataloader = dataloader(training_dataset, batch_size=tft_config.train_batch_size, train=True)
    val_dataloader = dataloader(validation_dataset, batch_size=tft_config.val_batch_size, train=False)
    
    
    tft = load_model(checkpoint_path)

    for param  in tft.parameters():
        param.requires_grad = False
    
    for name, module in tft.named_modules():
        if name in tft_config.layers_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
            logging.info(f"Unfroze layer: {name}")
    # Fine-tuning logic would go here (e.g., using PyTorch Lightning Trainer to fit the model on the new data)
    tft.hparams.learning_rate = tft_config.fine_tune_lr


    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=3, mode="min")
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{file_config.tft_models_dir}",
        filename=file_config.fine_tuned_name,
        save_top_k=1,
        mode="min"
    )
    wandb_logger = WandbLogger(project="pv-forecasting", name="tft-finetuning")
    trainer = Trainer(
        max_epochs = tft_config.fine_tune_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=tft_config.grad_clip_val,
        callbacks=[early_stop, checkpoint],
        logger=wandb_logger,
    )
    trainer.fit(tft, train_dataloader, val_dataloader)
    return tft, trainer

if __name__ == "__main__":
    wandb_login()
    os.makedirs(f"{file_config.tft_models_dir}", exist_ok=True) 
    data = load_data(file_config.fine_tuning_households)
    transfer_learning(data, file_config.tft_checkpoint_path)
    wandb.finish() #type: ignore 
    