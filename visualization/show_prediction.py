import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import image_size, tf_image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask
from pneumothorax_segmentation.unet import Unet

def show_prediction(folder, index):
    images_list = get_all_images_list(folder)

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    file_path, filename = images_list[index]
    dicom_data = get_dicom_data(file_path)

    # Display the final image
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.set_window_title("Show Predicted Mask of %s" % filename)

    plt.subplot(1, 2, 1)
    if (folder == "train"):
        true_mask = get_true_mask(filename)
        pixels = np.array(dicom_data.pixel_array) * (1 - .5 * true_mask)
        plt.imshow(pixels)
    else:
        plt.imshow(dicom_data.pixel_array)

    plt.subplot(1, 2, 2)
    unet = Unet()
    unet.load_weights("./weights/unet")
    image = tf.convert_to_tensor(dicom_data.pixel_array, dtype=tf.float32)
    image = tf.reshape(image, (1, image_size, image_size, 1))
    image = tf.image.resize(image, (tf_image_size, tf_image_size))
    predicted_logits = unet(image)
    predicted_logits = tf.image.resize(predicted_logits, (image_size, image_size))
    predicted_logits = predicted_logits.numpy()
    predictions = np.apply_along_axis(lambda l: np.argmax(l), axis=3, arr=predicted_logits)
    predictions = np.reshape(predictions, (image_size, image_size))
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

