from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_tf

def training_generator():
    "Yields a tuple (image, true_mask). Image is a tensor of shape (1, tf_image_size, tf_image_size, 1) and true_mask is a numpy array of shape (image_size, image_size)"
    images_list = get_all_images_list("train")
    for (filepath, filename) in images_list:
        dicom_data = get_dicom_data(filepath)
        image = format_pixel_array_for_tf(dicom_data.pixel_array)
        true_mask = get_true_mask(filename)

        yield (image, true_mask)
