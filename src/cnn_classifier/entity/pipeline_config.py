from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    params_image_size: list
    params_include_top: bool
    params_weights: str
    params_learning_rate: float
    params_classes: int

@dataclass
class CallbacksConfig:
    root_dir: Path
    tensorboard_log_dir: Path
    model_checkpoint_filepath: Path