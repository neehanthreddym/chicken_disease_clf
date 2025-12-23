from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.evaluation import Evaluation
from cnn_classifier import logger

STAGE_NAME = "Stage 04 - Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def execute_model_evaluation(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()

if __name__ == '__main__':
    try:
        logger.info(f">>> {STAGE_NAME}: started")
        model_evaluation_obj = ModelEvaluationPipeline()
        model_evaluation_obj.execute_model_evaluation()
        logger.info(f">>> {STAGE_NAME}: completed\n")
    except Exception as e:
        logger.exception(e)
        raise e
