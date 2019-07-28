import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

from pneumothorax_segmentation.preprocess import get_dicom_data, get_true_mask

def show_true_mask(index):
    dicom_data, filename = get_dicom_data("train", index) # True mask is only known for the training dataset
    mask = get_true_mask(filename.replace(".dcm", ""))

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
