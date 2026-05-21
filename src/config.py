
"""
Configuration file for hyperparameters and settings
"""
import os
from dataclasses import dataclass, field



@dataclass
class FileConfig:
    """File paths and related configurations."""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir: str = os.path.join(base_dir, "data/raw")
    processed_data_dir: str = os.path.join(base_dir, "data/processed")
    data_path: str = os.path.join(processed_data_dir, "pvgis_all.parquet")
    fine_tuning_households:str = os.path.join(processed_data_dir,"ukpv_london_tft.parquet")
    test_set:str = os.path.join(processed_data_dir, "ukpv_london_test_household.parquet")
    results_dir: str = os.path.join(base_dir, "results")
    models_dir: str = os.path.join(results_dir, "models")
    tft_models_dir: str = os.path.join(models_dir, "tft")
    tft_checkpoint_name: str = "tft-best-model.ckpt"
    tft_checkpoint_path: str = os.path.join(tft_models_dir, tft_checkpoint_name)
    fine_tuned_name: str = "tft-finetuned-model.ckpt"
    fine_tuned_path: str = os.path.join(tft_models_dir, fine_tuned_name)
    locations: list[dict[str, str | float]] = field(default_factory=lambda: [
        {"lat": 51.5,  "lon": -0.1,  "name": "London"},
        {"lat": 48.8,  "lon": 2.3,   "name": "Paris"},
        {"lat": 52.5,  "lon": 13.4,  "name": "Berlin"},
        {"lat": 40.4,  "lon": -3.7,  "name": "Madrid"},
        {"lat": 41.9,  "lon": 12.5,  "name": "Rome"},
        {"lat": 52.3,  "lon": 4.9,   "name": "Amsterdam"},
        {"lat": 38.7,  "lon": -9.1,  "name": "Lisbon"},
        {"lat": 52.2,  "lon": 21.0,  "name": "Warsaw"},
        {"lat": 59.3,  "lon": 18.1,  "name": "Stockholm"},
        {"lat": 45.7,  "lon": 4.8,   "name": "Lyon"},
    ])
@dataclass
class TFTConfig:
    """TFT Model hyperparameters and training settings."""
    
    train_batch_size: int = 1024
    val_batch_size: int = 1024
    test_batch_size: int = 1024
    max_encoder_length: int = 168
    max_prediction_length: int = 24
    hidden_size:int = 41
    attention_heads:int = 2
    dropout:float = 0.2
    hidden_continuous_size:int = 9
    grad_clip_val:float = 0.9
    seed:int = 42
    lr:float = 0.01
    num_workers:int = 4
    fine_tune_lr:float = 0.001
    fine_tune_epochs:int = 20
    finetune_batch_size:int = 256
    layers_to_unfreeze: list[str] = field(default_factory=lambda: [
        "output_layer",
        "pre_output_gate_norm",
        "static_enrichment",
        "decoder_variable_selection",
        "encoder_variable_selection",
    ])

@dataclass
class ResidualsConfig:
    """Residuals model hyperparameters and training settings."""
    batch_size: int = 32
    hidden_size: int = 32
    dropout: float = 0.1
    lr: float = 0.0001
    seed: int = 42




# Global configuration instances
file_config = FileConfig()
tft_config = TFTConfig()
residuals_config = ResidualsConfig()

