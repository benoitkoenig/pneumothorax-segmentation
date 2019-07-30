import numpy as np
import tensorflow as tf
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.segmentation.params import disease_pixel_weight

def calculate_loss(predicted_logits, true_mask):
    "Calculates loss. predicted_logits are the output of the Unet model. True_mask is of shape (image_size, image_size)"
    resized_predicted_logits = tf.image.resize(predicted_logits, (image_size, image_size))
    reshaped_true_mask = np.reshape(true_mask, (1, image_size, image_size))

    pixel_loss = sparse_softmax_cross_entropy_with_logits(logits=resized_predicted_logits, labels=reshaped_true_mask)
    pixel_loss = tf.multiply(disease_pixel_weight * reshaped_true_mask + 1, pixel_loss)

    return tf.reduce_sum(pixel_loss)
