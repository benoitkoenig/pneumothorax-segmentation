import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask

def show_true_mask(index):
    images_list = get_all_images_list("train")

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    file_path, filename = images_list[index]
    dicom_data = get_dicom_data(file_path) # True mask is only known for the training dataset
    mask = get_true_mask(filename)

    if np.max(mask) == 0:
        print("No Pneumothorax on %s" % filename)
        exit(0)

    pixels = np.array(dicom_data.pixel_array) * (1 - .5 * mask)

    # Display the final image
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.set_window_title("Show True Mask of %s" % filename)

    plt.imshow(pixels)

    plt.show()

# Read arguments from python command
index = None
for param in sys.argv:
    if param.isdigit():
        index = int(param)

if (index == None):
    print("Usage: python show_true_mask.py [id]")
    print("Hint: images 0, 5 and 22 have a mask")
    exit(-1)

show_true_mask(index)
