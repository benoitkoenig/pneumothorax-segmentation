from segmentation_models import Unet
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path, tf_image_size

def get_prediction(image):
    unet = Unet(
        'resnet34',
        encoder_weights='imagenet',
        classes=2,
        activation="linear",
        input_shape=(tf_image_size, tf_image_size, 3),
    )
    unet.load_weights(folder_path + "/weights/unet.hdf5")
    prediction = unet.predict(image, steps=1)
    return prediction
