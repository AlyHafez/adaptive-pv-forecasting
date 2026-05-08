
"""
Configuration file for hyperparameters and settings
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FileConfig:
    """File paths and related configurations."""
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
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    







# Global configuration instances
file_config = FileConfig()

