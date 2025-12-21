from cnn_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from cnn_classifier.pipeline.stage02_model_definition import ModelDefinitionPipeline
from cnn_classifier import logger

# Stage 01 - Data Ingestion Pipeline
STAGE_NAME = "Stage 01 - Data Ingestion"

try:
    logger.info(f">>> {STAGE_NAME}: started")
    data_ingestion_obj = DataIngestionPipeline()
    data_ingestion_obj.execute_data_ingestion()
    logger.info(f">>> {STAGE_NAME}: completed\n")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 02 - Model Definition Pipeline
STAGE_NAME = "Stage 02 - Model Definition"
try:
    logger.info(f">>> {STAGE_NAME}: started")
    model_definition_obj = ModelDefinitionPipeline()
    model_definition_obj.execute_model_definition()
    logger.info(f">>> {STAGE_NAME}: completed\n")
except Exception as e:
    logger.exception(e)
    raise e