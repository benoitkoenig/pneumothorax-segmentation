import matplotlib.pyplot as plt
import sys
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import image_size
import pneumothorax_segmentation.data_augment as data_augment
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data

def show_subplot(plt, index, image, title):
    "Show the image in subplot at the given index with the given title"
    plt.subplot(2, 4, index)
    img = tf.gather(image, 0)
    img = img / 255.
    img = tf.image.grayscale_to_rgb(img)
    plt.imshow(img)
    plt.title(title)

def show_data_augment(index):
    images_list = get_all_images_list("train")

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    file_path, filename = images_list[index]
    dicom_data = get_dicom_data(file_path)
    image = tf.convert_to_tensor(dicom_data.pixel_array, dtype=tf.float32)
    image = tf.reshape(image, (1, image_size, image_size, 1))

    # Display the final image
    fig = plt.figure(figsize=(18, 12))
    fig.canvas.set_window_title("Show data augment %s" % filename)

    show_subplot(plt, 1, image, "Original Image")
    show_subplot(plt, 2, data_augment.get_tall_image(image), "Tall Image")
    show_subplot(plt, 3, data_augment.get_large_image(image), "Large Image")
    show_subplot(plt, 4, data_augment.get_zoomed_image(image), "Zoomed Image")
    show_subplot(plt, 5, data_augment.get_mirror_image(image), "Mirror Image")
    show_subplot(plt, 6, data_augment.rotate_and_zoom_image(image), "Rotated and Zoomed Image")
    show_subplot(plt, 7, data_augment.smooth_image(image), "Smooth Image")
    show_subplot(plt, 8, data_augment.change_image_brightness(image), "Bright Image")

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
