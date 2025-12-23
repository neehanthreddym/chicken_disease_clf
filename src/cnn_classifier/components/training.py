import tensorflow as tf
from cnn_classifier.entity.pipeline_config import TrainingConfig
from pathlib import Path
import math

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_generator = None
        self.valid_generator = None
    
    def get_base_model(self):
        # Load WITHOUT optimizer/compile state
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path, compile=False)

        # Compile fresh optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
            steps_per_execution=10,
            jit_compile="auto"
        )
    
    def train_validation_generator(self):
        ds_kwargs = dict(
            directory=self.config.training_data,
            validation_split=0.20,
            seed=42,
            image_size=self.config.params_image_size[:-1],  # e.g., [224, 224]
            batch_size=self.config.params_batch_size,
            label_mode="categorical",
            interpolation="bilinear"
        )

        train_ds = tf.keras.utils.image_dataset_from_directory(
            subset="training",
            shuffle=True,
            **ds_kwargs
        )

        validation_ds = tf.keras.utils.image_dataset_from_directory(
            subset="validation",
            shuffle=False,
            **ds_kwargs
        )

        # Rescale
        layers = tf.keras.layers
        AUTOTUNE = tf.data.AUTOTUNE
        rescale = layers.Rescaling(1.0 / 255)
        
        # # Cache raw decoded images
        # train_ds = train_ds.cache()
        # validation_ds = validation_ds.cache()

        # Augmentation
        if self.config.params_is_augmentation:
            aug = tf.keras.Sequential([
                layers.RandomRotation(40 / 360),          # ~40 degrees
                layers.RandomFlip("horizontal"),
                layers.RandomTranslation(0.2, 0.2),       # height, width shift
                layers.RandomZoom(0.2, 0.2),
                layers.RandomShear(0.2, 0.2)              # shear transformations
            ])

            train_ds = train_ds.map(
                lambda x, y: (aug(rescale(x), training=True), y),
                num_parallel_calls=AUTOTUNE
            )
        else:
            train_ds = train_ds.map(
                lambda x, y: (rescale(x), y),
                num_parallel_calls=AUTOTUNE
            )
        
        validation_ds = validation_ds.map(
            lambda x, y: (rescale(x), y),
            num_parallel_calls=AUTOTUNE
        )

        # Performance
        train_ds = train_ds.cache().prefetch(AUTOTUNE)
        validation_ds = validation_ds.cache().prefetch(AUTOTUNE)

        # Assign to class attributes
        self.train_generator = train_ds
        self.valid_generator = validation_ds

        # Save number of images for steps calculation
        self.num_train_images = int((1 - ds_kwargs['validation_split']) * 8067)  # 80% split
        self.num_val_images = int(ds_kwargs['validation_split'] * 8067)

    def train(self, callback_list: list):
        self.steps_per_epoch = math.ceil(self.num_train_images / self.config.params_batch_size)
        self.validation_steps = math.ceil(self.num_val_images / self.config.params_batch_size)
    
        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Load the best model saved by ModelCheckpoint
        self.model = tf.keras.models.load_model(self.config.model_checkpoint_filepath)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)