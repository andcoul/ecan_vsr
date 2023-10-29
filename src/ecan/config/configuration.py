import os
from ecan.constants import *
from ecan.entity.config_entity import DataIngestionConfig, TrainingConfig
from ecan.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=Path(config.source_URL),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        training_data = os.path.join(self.config.data_ingestion.root_dir, 'dataset')
        create_directories([config.root_dir])

        data_ingestion_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=self.params.N_EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_scale_factor=self.params.SCALE_FACTOR,
            params_nFrame=self.params.N_FRAMES
        )

        return data_ingestion_config
