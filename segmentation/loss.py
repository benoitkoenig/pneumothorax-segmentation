import keras.backend as K
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.segmentation.params import mask_factor, non_mask_factor

def get_bce_loss(true_mask, predicted_logits):
    resized_predicted_logits = tf.image.resize(predicted_logits, (image_size, image_size))
    loss = K.binary_crossentropy(true_mask, resized_predicted_logits)
    loss = mask_factor * K.mean(true_mask * loss) + non_mask_factor * K.mean((1 - true_mask) * loss)
    return loss
