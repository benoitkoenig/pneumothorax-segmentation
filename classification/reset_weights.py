import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import tf_image_size
from pneumothorax_segmentation.classification.classifier import Classifier

def reset_weights():
    "Reset the weights of the Classifier model to random values"
    classifier = Classifier()

    random_image = tf.convert_to_tensor(np.random.random((1, tf_image_size, tf_image_size, 1)), dtype=tf.float32)

    _ = classifier(random_image)

    classifier.save_weights("./weights/classifier")

reset_weights()
