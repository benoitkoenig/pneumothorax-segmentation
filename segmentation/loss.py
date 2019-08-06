import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.segmentation.params import disease_pixel_weight

def calculate_loss(true_mask, predicted_logits):
    "Calculates loss. predicted_logits are the output of the Unet model. True_mask is of shape (1, image_size, image_size)"
    resized_predicted_logits = tf.image.resize(predicted_logits, (image_size, image_size))

    pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=resized_predicted_logits, labels=true_mask)
    pixel_loss = tf.multiply(disease_pixel_weight * tf.cast(true_mask, dtype=tf.float32) + 1, pixel_loss)

    return tf.reduce_sum(pixel_loss)
