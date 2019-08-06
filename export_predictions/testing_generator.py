from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, format_pixel_array_for_tf

def testing_generator(graph):
    """
    Yields an (image, filename) tuple from the test folder as a tensor of shape (1, tf_image_size, tf_image_size, 3)\n
    Due to the way generators work, it is required to specify the graph to work on 
    """
    images_list = get_all_images_list("test")

    for (filepath, filename) in images_list:
        with graph.as_default():
            dicom_data = get_dicom_data(filepath)
            image = format_pixel_array_for_tf(dicom_data.pixel_array)
        yield (image, filename)
