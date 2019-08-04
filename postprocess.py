import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size

def build_predicted_mask(predicted_logits):
    "Inputs the predicted logits as a list of shape (1, 256, 256, 2). Outputs a np matrix of shape (image_size, image_size)"
    predictions = tf.convert_to_tensor(predicted_logits, dtype=tf.float32)
    predictions = tf.image.resize(predicted_logits, (image_size, image_size))
    predictions = predictions.numpy()
    predictions = np.apply_along_axis(lambda l: np.argmax(l), axis=3, arr=predictions)
    predictions = np.reshape(predictions, (image_size, image_size))
    return predictions
