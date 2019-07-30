import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_tf
from pneumothorax_segmentation.segmentation.loss import calculate_loss
from pneumothorax_segmentation.segmentation.params import learning_rate
from pneumothorax_segmentation.segmentation.unet import Unet
from pneumothorax_segmentation.tracking import save_segmentation_data

def train():
    unet = Unet()
    unet.load_weights("./weights/unet")
    opt = Adam(learning_rate=learning_rate)

    images_list = get_all_images_list("train")
    for (index, (filepath, filename)) in enumerate(images_list):
        dicom_data = get_dicom_data(filepath)
        image = format_pixel_array_for_tf(dicom_data.pixel_array)
        true_mask = get_true_mask(filename)

        def get_loss():
            predicted_logits = unet(image)
            save_segmentation_data(index, predicted_logits, true_mask)
            return calculate_loss(predicted_logits, true_mask)

        opt.minimize(get_loss, [unet.trainable_weights])

        # if index % 100 == 99:
        #     unet.save_weights("./weights/unet")
        unet.save_weights("./weights/unet")

    print("Training done over %s images. Saving final weights" % len(images_list))
    unet.save_weights("./weights/unet")

train()
