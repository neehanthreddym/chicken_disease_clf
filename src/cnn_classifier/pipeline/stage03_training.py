from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.callbacks import Callbacks
from cnn_classifier.components.training import Training
from cnn_classifier import logger

STAGE_NAME = "Stage 03 - Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def execute_model_definition(self):
        config = ConfigurationManager()
        callbacks_config = config.get_callbacks_config()
        prepare_callbacks = Callbacks(config=callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_validation_generator()
        training.train(callback_list=callback_list)

if __name__ == '__main__':
    try:
        logger.info(f">>> {STAGE_NAME}: started")
        model_training_obj = ModelTrainingPipeline()
        model_training_obj.execute_model_definition()
        logger.info(f">>> {STAGE_NAME}: completed\n")
    except Exception as e:
        logger.exception(e)
        raise e
