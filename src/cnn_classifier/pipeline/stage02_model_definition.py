from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_definition import BaseModel
from cnn_classifier import logger

STAGE_NAME = "Stage 02 - Model Definition"

class ModelDefinitionPipeline:
    def __init__(self):
        pass

    def execute_model_definition(self):
        config = ConfigurationManager()
        base_model_confg = config.get_base_model_config()
        base_model = BaseModel(config=base_model_confg)
        base_model.get_base_model()
        base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f">>> {STAGE_NAME}: started")
        model_definition_obj = ModelDefinitionPipeline()
        model_definition_obj.execute_model_definition()
        logger.info(f">>> {STAGE_NAME}: completed\n")
    except Exception as e:
        logger.exception(e)
        raise e
