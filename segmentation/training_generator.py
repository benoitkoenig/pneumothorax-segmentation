import numpy as np
import random
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_tf

def get_training_generator():
    images_list_root = get_all_images_list("train")
    random.shuffle(images_list_root)
    def training_generator(graph, starting_index=0):
        """
        Yields a tuple (image, true_mask). Image is a tensor of shape (1, tf_image_size, tf_image_size, 3) and true_mask is a tensor of shape (1, image_size, image_size)\n
        Due to the way generators work, it is required to specify the graph to work on
        """
        images_list = images_list_root[starting_index:]
        for (filepath, filename) in images_list:
            with graph.as_default():
                dicom_data = get_dicom_data(filepath)
                image = format_pixel_array_for_tf(dicom_data.pixel_array)
                true_mask = get_true_mask(filename)
                true_mask = tf.convert_to_tensor(true_mask, dtype=tf.float32)
                true_mask = tf.reshape(true_mask, (1, image_size, image_size, 1))
            yield ([image], [true_mask])

    return training_generator
