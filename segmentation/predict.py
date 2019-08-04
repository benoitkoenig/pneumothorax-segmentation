import tensorflow as tf

from pneumothorax_segmentation.constants import  folder_path
from pneumothorax_segmentation.segmentation.unet import build_unet
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, format_pixel_array_for_tf

def get_prediction(image):
    unet = build_unet()
    unet.load_weights(folder_path + "/weights/unet.hdf5")
    return unet.predict(image, steps=1)
