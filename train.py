import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import image_size, tf_image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask
from pneumothorax_segmentation.tracking import save_data
from pneumothorax_segmentation.unet import Unet

def get_tf_image_and_mask(filepath, filename):
    "Return the tensorflow image of shape (1, tf_image_size, tf_image_size, 1) and the numpy mask of shape (1, image_size, image_size)"
    dicom_data = get_dicom_data(filepath)
    true_mask = get_true_mask(filename)

    image = tf.convert_to_tensor(dicom_data.pixel_array, dtype=tf.float32)
    image = tf.reshape(image, (1, image_size, image_size, 1))
    image = tf.image.resize(image, (tf_image_size, tf_image_size))

    true_mask = np.reshape(true_mask, (1, image_size, image_size))

    return image, true_mask

def train():
    unet = Unet()
    unet.load_weights("./weights/unet")
    opt = Adam(learning_rate=1e-4)

    images_list = get_all_images_list("train")
    for (index, (filepath, filename)) in enumerate(images_list):
        image, true_mask = get_tf_image_and_mask(filepath, filename)

        def get_loss():
            predicted_logits = unet(image)
            predicted_logits = tf.image.resize(predicted_logits, (image_size, image_size))
            save_data(index, predicted_logits.numpy(), true_mask)
            return sparse_softmax_cross_entropy_with_logits(logits=predicted_logits, labels=true_mask)

        opt.minimize(get_loss, [unet.trainable_weights])

        if index % 100 == 99:
            print("Index %s processed - Weights saved" % index)
            unet.save_weights("./weights/unet")
        else:
            print("Index %s processed" % index)

    print("Training done over %s images. Saving final weights" % len(images_list))
    unet.save_weights("./weights/unet")

train()
