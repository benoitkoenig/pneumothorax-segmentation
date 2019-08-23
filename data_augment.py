import random
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size

bigger_size = int(1.25 * image_size)
padding_for_big = 128

def get_tall_image(image):
    "Zooms on the image along the y-axis only"
    tall_image = tf.image.resize(image, (bigger_size, image_size), align_corners=True, method=tf.image.ResizeMethod.AREA)
    tall_image = tf.image.crop_to_bounding_box(tall_image, padding_for_big, 0, image_size, image_size)
    return tall_image

def get_large_image(image):
    "Zooms on the image along the x-axis only"
    large_image = tf.image.resize(image, (image_size, bigger_size), align_corners=True, method=tf.image.ResizeMethod.AREA)
    large_image = tf.image.crop_to_bounding_box(large_image, 0, padding_for_big, image_size, image_size)
    return large_image

def get_zoomed_image(image):
    "Zooms on the image"
    zoom_image = tf.image.resize(image, (bigger_size, bigger_size), align_corners=True, method=tf.image.ResizeMethod.AREA)
    zoom_image = tf.image.crop_to_bounding_box(zoom_image, padding_for_big, padding_for_big, image_size, image_size)
    return zoom_image

def get_mirror_image(image):
    "Flip the image left-right"
    mirror_image = tf.image.flip_left_right(image)
    return mirror_image

def rotate_and_zoom_image(image):
    "Rotates the image. This action is combined with zooming to avoid black edges. Angle is determined at random"
    angle = random.random() * .2 - .1
    rotated_image = tf.contrib.image.rotate(image, angle)
    big_rotated_image = get_zoomed_image(rotated_image) # Zooming is necessary to avoid black edges
    return big_rotated_image

def smooth_image(image):
    "Smoothes the image"
    smoothed_image = tf.image.resize(image, (256, 256), align_corners=True, method=tf.image.ResizeMethod.AREA)
    smoothed_image = tf.image.resize(smoothed_image, (image_size, image_size), align_corners=True)
    return smoothed_image

def change_image_brightness(image):
    "Darkens or brightens the image at random"
    delta = random.random() * .4 + .6
    if (random.randint(0, 1) == 1):
        dark_image = image * delta
        return dark_image
    else:
        bright_image = 255. - delta * (255. - image)
        return bright_image

def apply_random_data_augment(image, technique):
    """
        Apply a data augmentation technique picked at random among a certain pool of techniques. These pools are:\n
        'none': No data augmentation applied\n
        'resize': zooms along the x-axis, the y-axis, or both\n
        'flip_rotate': flips the image left-right with .5 prob and apply a slight random rotation\n
        'filter': smoothes or changes the image's brightness\n
        Image inputs are tensor of dim 3 or 4. Outputs are the same shape as inputs
    """
    if (technique == "none"):
        return image
    elif (technique == "resize"):
        rand = random.randint(0, 3)
        if (rand == 0):
            return image
        if (rand == 1):
            return get_tall_image(image)
        if (rand == 2):
            return get_large_image(image)
        return get_zoomed_image(image)
    elif (technique == "flip_rotate"):
        rand = random.randint(0, 1)
        if (rand == 0):
            return rotate_and_zoom_image(image)
        else:
            mirror_image = get_mirror_image(image)
            return rotate_and_zoom_image(mirror_image)
    elif (technique == "filter"):
        if random.randint(0, 1) == 1:
            return smooth_image(image)
        else:
            return change_image_brightness(image)
    else:
        print("random_data_augment called with invalid technique. Exiting immediately")
        exit(-1)
