import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.classification.classifier import Classifier
from pneumothorax_segmentation.classification.params import learning_rate
from pneumothorax_segmentation.preprocess import get_all_images_list, get_dicom_data, get_true_mask, format_pixel_array_for_tf

def train():
    classifier = Classifier()
    classifier.load_weights(folder_path + "/weights/classifier")
    opt = Adam(learning_rate=learning_rate)

    images_list = get_all_images_list("train")
    for (index, (filepath, filename)) in enumerate(images_list):
        dicom_data = get_dicom_data(filepath)
        image = format_pixel_array_for_tf(dicom_data.pixel_array)
        true_mask = get_true_mask(filename)
        is_there_pneumothorax = np.max(true_mask) # 1 if there is a mask, 0 otherwise

        def get_loss():
            logits = classifier(image)
            print(index, is_there_pneumothorax, logits.numpy()[0])
            return sparse_softmax_cross_entropy_with_logits(logits=logits, labels=[is_there_pneumothorax])

        opt.minimize(get_loss, [classifier.trainable_weights])

        if index % 100 == 99:
            classifier.save_weights(folder_path + "/weights/classifier")

    print("Training done over %s images. Saving final weights" % len(images_list))
    classifier.save_weights(folder_path + "/weights/classifier")

train()
