import csv
import numpy as np
import pydicom
import os

from pneumothorax_segmentation.constants import image_size

# Documentation for reading dicom files at https://pydicom.github.io/pydicom/stable/viewing_images.html#using-pydicom-with-matplotlib

def get_dicom_data(folder, index):
    # Load all images filenames in folder
    all_images_in_folder = []

    for dirName, _, fileList in os.walk("./dicom-images-%s" % folder):
        for filename in fileList:
            if ".dcm" in filename.lower():
                all_images_in_folder.append((os.path.join(dirName,filename), filename))

    # Check index is valid
    if index >= len(all_images_in_folder):
        print("Index %s out of range. Max index is %s" % (index, len(all_images_in_folder) - 1))
        exit(-1)

    # Display the data and image through matplotlib
    ds = pydicom.dcmread(all_images_in_folder[index][0])
    return ds, all_images_in_folder[index][1]

def get_true_mask(name):
    # Retrieve masks as they are in the csv
    raw_masks = []
    with open('train-rle.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
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
    mask_mapping = np.zeros(image_size ** 2)
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
