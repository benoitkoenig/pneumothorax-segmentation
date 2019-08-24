import numpy as np
import random
import tensorflow as tf

from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_image_label, format_pixel_array_for_tf

def get_training_generator():
    images_list_root = get_all_images_list("train")
    random.shuffle(images_list_root)
    def training_generator(graph, data_augment_technique="none", starting_index=0):
        """
        Yields a tuple (image, is_there_pneumothorax). Image is a tensor of shape (1, tf_image_size, tf_image_size, 3) and true_mask is a tensor of shape (1, 2)\n
        Due to the way generators work, it is required to specify the graph to work on\n
        data_augment_technique is the name of the technique used by data_augmentation (see data_augment.py -> random_data_augment)
        """
        images_list = images_list_root[starting_index:]
        for (filepath, filename) in images_list:
            with graph.as_default():
                dicom_data = get_dicom_data(filepath)
                image = dicom_data.pixel_array
                image = format_pixel_array_for_tf(image, apply_data_augment_technique=data_augment_technique)
                is_there_pneumothorax = get_image_label(filename)
                is_there_pneumothorax = [[1 - is_there_pneumothorax, is_there_pneumothorax]]
                is_there_pneumothorax = tf.convert_to_tensor(is_there_pneumothorax, dtype=tf.float32)
            yield ([image], [is_there_pneumothorax])

    return training_generator
