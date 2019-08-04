import numpy as np

from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_tf

def training_generator():
    "Yields a tuple (image, is_there_pneumothorax). Image is a tensor of shape (1, tf_image_size, tf_image_size, 1) and is_there_pneumothorax is 0 or 1"
    images_list = get_all_images_list("train")
    for (filepath, filename) in images_list:
        dicom_data = get_dicom_data(filepath)
        image = format_pixel_array_for_tf(dicom_data.pixel_array)
        true_mask = get_true_mask(filename)
        is_there_pneumothorax = np.max(true_mask) # 1 if there is a mask, 0 otherwise

        yield (image, is_there_pneumothorax)
