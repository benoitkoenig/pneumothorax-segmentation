import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_tf

images_list = get_all_images_list("train") # Global for generator_length. It woud be better defined inside training_generator otherwise

generator_length = len(images_list)

def training_generator(graph):
    """
    Yields a tuple (image, true_mask). Image is a tensor of shape (1, tf_image_size, tf_image_size, 1) and true_mask is a tuple of shape (1, image_size, image_size)\n
    Due to the way generators work, it is required to specify the graph to work on 
    """
    for (filepath, filename) in images_list:
        with graph.as_default():
            dicom_data = get_dicom_data(filepath)
            image = format_pixel_array_for_tf(dicom_data.pixel_array)
            image = tf.image.grayscale_to_rgb(image)
            true_mask = get_true_mask(filename)
            true_mask = tf.convert_to_tensor(true_mask, dtype=tf.int32)
            true_mask = tf.reshape(true_mask, (1, image_size, image_size))
        yield ([image], [true_mask])
