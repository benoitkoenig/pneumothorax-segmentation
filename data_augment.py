import numpy as np
import tensorflow as tf

from constants import image_size

bigger_size = int(1.25 * image_size)

def get_mirror_image(image):
    return np.flip(image, axis=1)

def get_sized_image(input_image):
    image = np.reshape(input_image, (image_size, image_size, 1)) # Reshape needs to be done so tf resize is possible
    image = tf.convert_to_tensor(image) # I use tf methods for resizing, even though the output is still numpy

    tall_image = tf.image.resize(image, (bigger_size, image_size))
    tall_image = tf.image.crop_to_bounding_box(tall_image, 128, 0, 1024, 1024)
    tall_image = tall_image.numpy()
    tall_image = np.reshape(tall_image, (image_size, image_size))

    large_image = tf.image.resize(image, (image_size, bigger_size))
    large_image = tf.image.crop_to_bounding_box(large_image, 0, 128, 1024, 1024)
    large_image = large_image.numpy()
    large_image = np.reshape(large_image, (image_size, image_size))

    big_image = tf.image.resize(image, (bigger_size, bigger_size))
    big_image = tf.image.crop_to_bounding_box(big_image, 128, 128, 1024, 1024)
    big_image = big_image.numpy()
    big_image = np.reshape(big_image, (image_size, image_size))

    return [input_image, tall_image, large_image, big_image]

def get_many_images_from_one(image, mask):
    mirror_image = np.flip(image, axis=1)
    mirror_mask = np.flip(mask, axis=1)

    all_images = get_sized_image(image) + get_sized_image(mirror_image)
    all_masks = get_sized_image(mask) + get_sized_image(mirror_mask)

    return all_images, all_masks
