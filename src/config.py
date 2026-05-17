
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
    results_dir: str = os.path.join(base_dir, "results")
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

class ModelConfig:
    """Model hyperparameters and training settings."""
    train_batch_size: int = 64
    val_batch_size: int = 256
    test_batch_size: int = 256
    max_encoder_length: int = 168
    max_prediction_length: int = 24
    hidden_size:int = 32
    attention_heads:int = 4
    dropout:float = 0.1
    hidden_continuous_size:int = 16
    seed:int = 42
    lr:float = 0.01







# Global configuration instances
file_config = FileConfig()
model_config = ModelConfig()

