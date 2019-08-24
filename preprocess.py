import csv
import numpy as np
import os
import pydicom
from segmentation_models.backbones import get_preprocessing
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size, folder_path
from pneumothorax_segmentation.data_augment import apply_random_data_augment
from pneumothorax_segmentation.params import tf_image_size

# Documentation for reading dicom files at https://pydicom.github.io/pydicom/stable/viewing_images.html#using-pydicom-with-matplotlib

preprocess_input = get_preprocessing("resnet34")

def get_all_images_list(folder):
    "Load all images filenames in folder. Returns a list of (filepath, filename)"
    all_images_in_folder = []

    for dirName, _, fileList in os.walk(folder_path + "/data/dicom-images-%s" % folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                all_images_in_folder.append((os.path.join(dirName,filename), filename.replace(".dcm", "")))    

    return all_images_in_folder

def get_dicom_data(file_path):
    "Return the dicom raw data of a given file"
    return pydicom.dcmread(file_path)

cached_csv = []
def get_raw_masks(name):
    """
        Returns a list of the masks as they appear in train-rle.csv. Masks '-1' are filtered out\n
        Note side-effect: loads the csv on the first run and caches it
    """
    global cached_csv

    # The csv data is stored in a cache. This way, the csv is read only once
    if (len(cached_csv) == 0):
        with open(folder_path + '/data/train-rle.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                cached_csv.append(row)

    # Retrieve masks as they are in the csv
    raw_masks = []
    for row in cached_csv:
        if row[0] == name:
            raw_masks.append(row[1])

    # Remove the -1 from images with no mask
    if (raw_masks[0] == " -1"):
        raw_masks = []

    return raw_masks

def get_image_label(name):
    "Returns 1 if there is a pneumothorax, 0 otherwise. Based on data in train-rle.csv"
    raw_masks = get_raw_masks(name)
    if len(raw_masks) == 0:
        return 0
    return 1

def get_true_mask(name):
    "Takes the name of the image as input and returns the mask mapping as a numpy matrix of shape (image_size, image_size) and values 0-1"

    raw_masks = get_raw_masks(name)

    # Format the masks to an exploitable format
    masks = []
    for raw_mask in raw_masks:
        mask = raw_mask.split(" ")
        mask = mask[1:] # raw_mask starts with a space
        mask = [int(m) for m in mask]
        masks.append(mask)

    # Use the masks to create the actual mapping of image_size * image_size
    mask_mapping = np.zeros(image_size ** 2, dtype=np.int)
    for mask in masks:
        is_it_a_mask = False
        current_pixel = 0
        for pixel_long_movement in mask:
            if is_it_a_mask:
                for i in range(pixel_long_movement):
                    mask_mapping[current_pixel + i] = 1
            current_pixel += pixel_long_movement
            is_it_a_mask = not is_it_a_mask
    mask_mapping = np.reshape(mask_mapping, (image_size, image_size))
    mask_mapping = np.transpose(mask_mapping, (1, 0))

    return mask_mapping

def format_pixel_array_for_tf(pixel_array, apply_data_augment_technique=None):
    """
        Inputs pixel_array as they are stroed in the dicom file. Outputs a tensor ready to go through the models\n
        apply_data_augment_technique can be used to apply data augmentation. See apply_random_data_augment for values
    """
    image = tf.convert_to_tensor(pixel_array, dtype=tf.float32)
    image = tf.reshape(image, (1, image_size, image_size, 1))
    if (apply_data_augment_technique != None):
        image = apply_random_data_augment(image, apply_data_augment_technique)
    # tf.image.resize behaves weirdly with the default method when reducing size. AREA method makes more sense in our case, thought the default bilinear method makes more sense when making an image bigger
    image = tf.image.resize(image, (tf_image_size, tf_image_size), align_corners=True, method=tf.image.ResizeMethod.AREA)
    image = tf.image.grayscale_to_rgb(image)
    image = preprocess_input(image)
    return image
