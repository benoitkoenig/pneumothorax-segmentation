import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.data_augment import get_many_images_from_one
from pneumothorax_segmentation.preprocess import get_dicom_data, get_true_mask

def show_data_augment(index):
    dicom_data, filename = get_dicom_data("train", index) # True mask is only known for the training dataset
    image = np.array(dicom_data.pixel_array)
    mask = get_true_mask(filename.replace(".dcm", ""))

    images, masks = get_many_images_from_one(image, mask)

    # Display the final image
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.set_window_title("Show data augment %s" % filename)

    for i in range(8):
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(images[i])
        plt.subplot(4, 4, 2 * i + 2)
        plt.imshow(masks[i])

    plt.show()

# Read arguments from python command
index = None
for param in sys.argv:
    if param.isdigit():
        index = int(param)

if (index == None):
    print("Usage: python show_data_augment.py [id]")
    print("Hint: images 0, 5 and 22 have a mask")
    exit(-1)

show_data_augment(index)
