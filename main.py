from cnn_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from cnn_classifier import logger

STAGE_NAME = "Stage 01 - Data Ingestion"

try:
    logger.info(f">>> {STAGE_NAME}: started")
    data_ingestion_obj = DataIngestionPipeline()
    data_ingestion_obj.execute_data_ingestion()
    logger.info(f">>> {STAGE_NAME}: completed\n")
except Exception as e:
    logger.exception(e)
    raise e