import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import image_size, tf_image_size
from pneumothorax_segmentation.params import disease_pixel_weight, learning_rate
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_unet
from pneumothorax_segmentation.tracking import save_data
from pneumothorax_segmentation.unet import Unet

def train():
    unet = Unet()
    unet.load_weights("./weights/unet")
    opt = Adam(learning_rate=learning_rate)

    images_list = get_all_images_list("train")
    for (index, (filepath, filename)) in enumerate(images_list):
        dicom_data = get_dicom_data(filepath)
        true_mask = get_true_mask(filename)

        image = format_pixel_array_for_unet(dicom_data.pixel_array)
        true_mask = np.reshape(true_mask, (1, image_size, image_size))

        def get_loss():
            predicted_logits = unet(image)

            save_data(index, predicted_logits, true_mask)

            resized_predicted_logits = tf.image.resize(predicted_logits, (image_size, image_size))
            pixel_loss = sparse_softmax_cross_entropy_with_logits(logits=resized_predicted_logits, labels=true_mask)
            pixel_loss = tf.multiply(disease_pixel_weight * true_mask, pixel_loss)

            return tf.reduce_sum(pixel_loss)

        opt.minimize(get_loss, [unet.trainable_weights])

        if index % 100 == 99:
            print("Index %s processed - Weights saved" % index)
            unet.save_weights("./weights/unet")
        else:
            print("Index %s processed" % index)

    print("Training done over %s images. Saving final weights" % len(images_list))
    unet.save_weights("./weights/unet")

train()
