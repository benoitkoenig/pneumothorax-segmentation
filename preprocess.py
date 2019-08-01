import csv
import numpy as np
import os
import pydicom
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size, tf_image_size, folder_path

# Documentation for reading dicom files at https://pydicom.github.io/pydicom/stable/viewing_images.html#using-pydicom-with-matplotlib

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
def get_true_mask(name):
    "Takes the name of the image as input and returns the mask mapping as a numpy matrix of shape (image_size, image_size) and values 0-1"
    # Warning hidden side effect: get-true-mask loads the csv on the first run and caches it
    global cached_csv

    # The csv data is stored in a cache. This way, the csv is read only once
    if (len(cached_csv) == 0):
        with open('data/train-rle.csv') as csv_file:
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

def format_pixel_array_for_tf(pixel_array):
    "Inputs pixel_array as they are stroed in the dicom file. Outputs a tensor ready to go through the models"
    image = tf.convert_to_tensor(pixel_array, dtype=tf.float32)
    image = tf.reshape(image, (1, image_size, image_size, 1))
    image = tf.image.resize(image, (tf_image_size, tf_image_size))
    image = image / 255.
    return image
