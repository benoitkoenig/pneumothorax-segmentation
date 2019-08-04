import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.segmentation.loss import calculate_loss
from pneumothorax_segmentation.segmentation.params import learning_rate
from pneumothorax_segmentation.segmentation.unet import Unet
from pneumothorax_segmentation.segmentation.training_generator import training_generator
from pneumothorax_segmentation.tracking import save_segmentation_data

def train():
    unet = Unet()
    unet.load_weights(folder_path + "/weights/unet")
    opt = Adam(learning_rate=learning_rate)

    index = 0
    for (image, true_mask) in training_generator():
        def get_loss():
            predicted_logits = unet(image)
            save_segmentation_data(index, predicted_logits, true_mask)
            return calculate_loss(predicted_logits, true_mask)

        opt.minimize(get_loss, [unet.trainable_weights])

        index += 1
        if index % 100 == 99:
            unet.save_weights(folder_path + "/weights/unet")

    print("Training done over %s images. Saving final weights" % index)
    unet.save_weights(folder_path + "/weights/unet")

train()
