import tensorflow as tf
import tensorflow.keras.backend as K

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.segmentation.params import mask_factor, non_mask_factor

# Different loss functions available, gotta test and use the most efficient one

def get_bce_loss(true_mask, predicted_probs):
    "Calculates the weighted bce loss"
    loss = K.binary_crossentropy(true_mask, predicted_probs)
    anti_mask = 1 - true_mask
    true_mask_sum = K.sum(true_mask) + K.epsilon()
    anti_mask_sum = K.sum(anti_mask) + K.epsilon()
    loss = mask_factor * K.sum(true_mask * loss) / true_mask_sum + non_mask_factor * K.sum(anti_mask * loss) / anti_mask_sum
    return loss

def get_dice_loss(true_mask, predicted_probs):
    "Calculates the weighted dice loss. TODO: this loss function does not work properly. Please fix"
    labels = 2 * true_mask - 1
    predictions = 2 * predicted_probs - 1

    intersection = labels * predictions
    sum_of_each = K.square(labels) + K.square(predictions)

    intersection = true_mask * mask_factor * intersection + (1 - true_mask) * non_mask_factor * intersection
    sum_of_each = true_mask * mask_factor * sum_of_each + (1 - true_mask) * non_mask_factor * sum_of_each

    loss = 1 - 2 * K.sum(intersection) / (K.sum(sum_of_each) + K.epsilon())

    return loss

def calculate_loss(true_mask, predicted_probs):
    """
    Calculates the loss. true_mask should be of shape (image_size, image_size, ?) or
    (?, image_size, image_size, ?). predicted_probs will be resized to image_size x image_size
    """
    resized_predicted_probs = tf.image.resize(predicted_probs, (image_size, image_size), align_corners=True)
    return get_bce_loss(true_mask, resized_predicted_probs)
