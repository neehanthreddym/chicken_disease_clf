import tensorflow as tf
from pathlib import Path
from cnn_classifier.utils.utilities import save_json
from cnn_classifier.entity.pipeline_config import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.valid_generator = None
    
    def _validation_generator(self):
        validation_ds = tf.keras.utils.image_dataset_from_directory(
            subset="validation",
            shuffle=False,
            directory=self.config.training_data,
            validation_split=0.20,
            seed=42,
            image_size=self.config.params_image_size[:-1],  # e.g., [224, 224]
            batch_size=self.config.params_batch_size,
            label_mode="categorical",
            interpolation="bilinear"
        )

        self.class_names = validation_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE
        rescale = tf.keras.layers.Rescaling(1.0 / 255)

        validation_ds = validation_ds.map(
            lambda x, y: (rescale(x), y),
            num_parallel_calls=AUTOTUNE
        )
        validation_ds = validation_ds.cache().prefetch(AUTOTUNE)
        self.valid_generator = validation_ds
    
    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self._validation_generator()
        self.score = self.model.evaluate(self.valid_generator)
    
    def save_score(self):
        scores = scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "class_names": list(self.class_names)
        }
        save_json(path=Path("evaluation_scores.json"), data=scores)
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)