from pytorch_forecasting.models.temporal_fusion_transformer.tuning import  optimize_hyperparameters # type: ignore[import]
from torch.utils.data import DataLoader # type: ignore[import]
from pytorch_forecasting import TimeSeriesDataSet# type: ignore[import]
from src.models.dataset import split_data, create_dataset, dataloader# type: ignore[import]
from src.config import file_config, model_config# type: ignore[import]
import pandas as pd# type: ignore[import]
import optuna  # type: ignore[import]
import logging# type: ignore[import]
import pickle# type: ignore[import]
logging.basicConfig(level=logging.INFO)


def optimize_tft(train_dataloader, val_dataloader, model_path: str):
    """Optimize hyperparameters for the TFT model using Optuna."""
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path=model_path,
        n_trials=10,
        max_epochs=5,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 64),
        attention_head_size_range=(1, 4),
        dropout_range=(0.1, 0.3),
        learning_rate=model_config.lr,
        use_learning_rate_finder=False,
        trainer_kwargs=dict(
            limit_train_batches=30,
            accelerator="auto",
            devices=1,
            enable_progress_bar=False,
        ),
        reduce_on_plateau_patience=2,
    )
    return study 

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    data = pd.read_parquet(file_config.data_path)
    train_data, val_data, test_data = split_data(data)
    training_dataset = create_dataset(train_data, model_config.max_encoder_length, model_config.max_prediction_length)
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_data, stop_randomization=True)
    train_dataloader = dataloader(training_dataset, batch_size=model_config.train_batch_size, train=True)
    val_dataloader = dataloader(validation_dataset, batch_size=model_config.val_batch_size, train=False)
    
    study = optimize_tft(train_dataloader, val_dataloader, model_path=f"{file_config.results_dir}/optimized_tft.pkl")
    
    with open(f"{file_config.results_dir}/optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)
    logging.info("Optimization complete. Best hyperparameters saved to optimized_tft.pkl and study saved to optuna_study.pkl.")
    logging.info(f"Optimization complete. Best hyperparameters: {study.best_trial.params}")