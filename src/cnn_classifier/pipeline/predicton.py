from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

class PredictionPipeline:
    def __init__(self):
        # Load the trained model once during initialization
        self.model = load_model(os.path.join("artifacts", "training", "trained_model.h5"))
    
    def predict(self, image_input):
        if isinstance(image_input, str):
            test_image = image.load_img(image_input, target_size=(224, 224))
        else:
            test_image = image_input
            if test_image.size != (224, 224):
                test_image = test_image.resize((224, 224))

        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        result = np.argmax(self.model.predict(test_image), axis=1)
        print(f"Result = {result}")

        if result[0] == 0:
            return "Coccidiosis"
        elif result[0] == 1:
            return "Healthy"
        elif result[0] == 2:
            return "New Castle Disease"
        elif result[0] == 3:
            return "Salmonella"