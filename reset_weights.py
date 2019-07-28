import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.unet import Unet

def reset_weights():
    "Reset the weights of the Unet model to random values"
    unet = Unet()

    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 1)), dtype=tf.float32)

    _ = unet(random_image)

    unet.save_weights("./weights/unet")

reset_weights()
