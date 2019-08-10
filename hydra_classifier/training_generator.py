import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_image_label, format_pixel_array_for_tf

def training_generator(graph):
    """
    Yields a tuple (image, is_there_pneumothorax). Image is a tensor of shape (1, tf_image_size, tf_image_size, 3) and true_mask is a tensor of shape (1, 2)\n
    Due to the way generators work, it is required to specify the graph to work on 
    """
    images_list = get_all_images_list("train")
    for (filepath, filename) in images_list:
        with graph.as_default():
            dicom_data = get_dicom_data(filepath)
            image = format_pixel_array_for_tf(dicom_data.pixel_array)
            is_there_pneumothorax = get_image_label(filename)
            is_there_pneumothorax = [[1 - is_there_pneumothorax, is_there_pneumothorax]]
            is_there_pneumothorax = tf.convert_to_tensor(is_there_pneumothorax, dtype=tf.float32)
        yield ([image], [is_there_pneumothorax])
