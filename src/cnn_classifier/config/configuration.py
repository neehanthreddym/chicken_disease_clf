import os
from cnn_classifier.constants import *
from cnn_classifier.utils.utilities import read_yaml, create_directories
from cnn_classifier.entity.pipeline_config import (
    DataIngestionConfig,
    BaseModelConfig,
    CallbacksConfig
)

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        root_dir = Path(config.root_dir)
        local_data_file = Path(config.local_data_file)
        unzip_dir = Path(config.unzip_dir)

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=root_dir,
            source_URL=config.source_URL,
            local_data_file=local_data_file,
            unzip_dir=unzip_dir
        )

        return data_ingestion_config
    
    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_model_path= Path(config.updated_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
        )
        return base_model_config
    
    def get_callbacks_config(self) -> CallbacksConfig:
        config = self.config.callbacks
        model_ckpt_dir = os.path.dirname(config.model_checkpoint_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_log_dir)
        ])

        callbacks_config = CallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_log_dir=Path(config.tensorboard_log_dir),
            model_checkpoint_filepath=Path(config.model_checkpoint_filepath)
        )

        return callbacks_config