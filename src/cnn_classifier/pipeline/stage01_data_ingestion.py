from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_ingestion import DataIngestion
from cnn_classifier import logger

STAGE_NAME = "Stage 01 - Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def execute_data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_cofig = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_cofig)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()
        data_ingestion.organize_images_into_class_folders()
        data_ingestion.print_data_summary()

if __name__ == '__main__':
    try:
        logger.info(f">>> {STAGE_NAME}: started")
        data_ingestion_obj = DataIngestionPipeline()
        data_ingestion_obj.execute_data_ingestion()
        logger.info(f">>> {STAGE_NAME}: completed\n")
    except Exception as e:
        logger.exception(e)
        raise e