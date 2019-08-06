import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.classification.classifier import Classifier
from pneumothorax_segmentation.classification.params import learning_rate
from pneumothorax_segmentation.classification.training_generator import training_generator

def train():
    classifier = Classifier()
    classifier.load_weights(folder_path + "/weights/classifier")
    opt = Adam(learning_rate=learning_rate)

    index = 0
    for (image, is_there_pneumothorax) in training_generator():
        def get_loss():
            logits = classifier(image)
            return sparse_softmax_cross_entropy_with_logits(logits=logits, labels=[is_there_pneumothorax])

        opt.minimize(get_loss, [classifier.trainable_weights])

        index += 1
        if index % 100 == 99:
            classifier.save_weights(folder_path + "/weights/classifier")

    print("Training done over %s images. Saving final weights" % index)
    classifier.save_weights(folder_path + "/weights/classifier")

train()
