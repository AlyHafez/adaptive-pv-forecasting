from src.models.tft import create_tft, create_network
from src.models.dataset import dataloader, create_dataset, split_data
from src.data.data_config import file_config, model_config
from pytorch_forecasting.tuning import Tuner # type: ignore[import]
from pytorch_forecasting import TimeSeriesDataSet # type: ignore[import]
import matplotlib.pyplot as plt # type: ignore[import]
import wandb # type: ignore[import]
from dotenv import load_dotenv # type: ignore[import]
import logging
import os
import pandas as pd # type: ignore[import]
logging.basicConfig(level=logging.INFO)
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key is None:
    msg = "WANDB_API_KEY not found in environment variables. Please set it in the .env file."
    logging.error(msg)
    raise ValueError(msg)
else:
    logging.info("WANDB_API_KEY successfully loaded from environment variables.")
    wandb.login(key=wandb_api_key) #  type: ignore[attr-defined]


def find_learning_rate(trainer, model, train_dataloader):
    tuner = Tuner(trainer)
    res = tuner.lr_find(model, train_dataloader, min_lr=1e-5, max_lr=1, num_training=200)
    logging.info(f"Suggested learning rate: {res.suggestion()}")
    return res.results



    
def plot_lr_finder(results):
    wandb.init(project="pv-forecasting", name="lr_finder")
    
    for lr, loss in zip(results["lr"], results["loss"]):
        wandb.log({"lr": lr, "loss": loss})
    
    wandb.finish()

if __name__ == "__main__":
    trainer = create_network()
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, test_data = split_data(data)
    training_dataset = create_dataset(train_data, model_config.max_encoder_length, model_config.max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    train_dataloader = dataloader(training_dataset, batch_size=model_config.train_batch_size)
    val_dataloader = dataloader(validation_dataset, batch_size=model_config.val_batch_size, train=False)
    model = create_tft(training_dataset=training_dataset)  # Pass the training dataset since we need the model architecture for lr_find
    lr_finder_results = find_learning_rate(trainer, model, train_dataloader)
    lr_df = pd.DataFrame(lr_finder_results)
    plot_lr_finder(lr_df)
