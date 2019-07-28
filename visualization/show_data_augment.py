import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.data_augment import get_many_images_from_one
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask

def show_data_augment(index):
    images_list = get_all_images_list("train")

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    file_path, filename = images_list[index]
    dicom_data = get_dicom_data(file_path)

    image = np.array(dicom_data.pixel_array)
    mask = get_true_mask(filename)

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
