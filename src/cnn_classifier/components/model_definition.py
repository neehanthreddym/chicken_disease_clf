import tensorflow as tf
from cnn_classifier.entity.pipeline_config import BaseModelConfig
from pathlib import Path

class BaseModel:
    def __init__(self, config = BaseModelConfig):
        self.config=config
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )
    
    @staticmethod
    def define_full_model(model, classes, freez_all, freeze_till):
        if freez_all:
            for layer in model.layers:
                model.trainable = False
        if (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
            
        flatten_layer = tf.keras.layers.Flatten()(model.output)
        output_layer = tf.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_layer)

        full_model = tf.keras.Model(
            inputs=model.input,
            outputs=output_layer
        )

        # full_model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     metrics=["accuracy"]
        # )
        full_model.summary()

        return full_model
    
    def update_base_model(self):
        self.full_model = self.define_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freez_all=True,
            freeze_till=None
            # learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_model_path,
            model=self.full_model
        )
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        return model.save(path)