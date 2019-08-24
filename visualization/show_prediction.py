import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask
from pneumothorax_segmentation.segmentation.predict import get_segmentation_prediction

def show_prediction(folder, index):
    images_list = get_all_images_list(folder)

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    filepath, filename = images_list[index]
    dicom_data = get_dicom_data(filepath)

    # Display the final image
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.set_window_title("Show Predicted Mask of %s" % filename)

    plt.subplot(1, 2, 1)
    pixels = dicom_data.pixel_array
    if (folder == "train"):
        true_mask = get_true_mask(filename)
        pixels = np.array(pixels) * (1 - .5 * true_mask)
    plt.imshow(pixels)

    plt.subplot(1, 2, 2)
    predictions = get_segmentation_prediction(filepath)

    plt.imshow(predictions)

    plt.show()

# Read arguments from python command
folder = None
index = None
for param in sys.argv:
    if param in ["train", "test"]:
        folder = param
    if param.isdigit():
        index = int(param)

if (index == None):
    print("Usage: python show_prediction.py [id]")
    print("Hint: images 0, 5 and 22 have a mask")
    exit(-1)
if (folder == None):
    print("No folder specified. Defaults to train")
    folder = "train"

show_prediction(folder, index)
