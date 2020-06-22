import numpy as np
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


_IMAGE_NET_TARGET_SIZE = (224, 224)


class Img2Vec(object):
    """Transform an image to a vector."""
    def __init__(self):
        model = resnet50.ResNet50(weights='imagenet')
        layer_name = 'avg_pool'
        self.intermediate_layer_model = Model(inputs=model.input,
                                              outputs=model.get_layer(layer_name).output)

    def get_vec(self, image_path):
        """
        Gets a vector embedding from an image.

        Parameters
        -------------
        image_path: str or Path
            path to image on filesystem

        Returns
        -------------
        embedding: np.arrays

        """
        img = image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        intermediate_output = self.intermediate_layer_model.predict(x)

        return intermediate_output[0]
