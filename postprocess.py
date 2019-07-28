import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size

def build_predicted_mask(predicted_logits):
    "Inputs the predicted logits as they are returned bu the Unet model. Outputs a np matrix of shape (image_size, image_size, 1)"
    predictions = tf.image.resize(predicted_logits, (image_size, image_size))
    predictions = predictions.numpy()
    predictions = np.apply_along_axis(lambda l: np.argmax(l), axis=3, arr=predictions)
    predictions = np.reshape(predictions, (image_size, image_size))
    return predictions
