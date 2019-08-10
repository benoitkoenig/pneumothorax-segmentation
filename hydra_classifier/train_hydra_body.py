from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import sys
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.hydra_classifier.get_classifier import get_classifier
from pneumothorax_segmentation.hydra_classifier.params import learning_rate, steps_per_epoch_body, epochs_body
from pneumothorax_segmentation.hydra_classifier.training_generator import training_generator

# Hydra: an Ensemble of Convolutional NeuralNetworks for Geospatial Land Classification: https://arxiv.org/pdf/1802.03518.pdf

graph = tf.compat.v1.get_default_graph()

def train_hydra_body(backbone_name):
    model = get_classifier(backbone_name)
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")

    model_checkpoint = ModelCheckpoint(folder_path + "/weights/hydra_%s_body.hdf5" % backbone_name)

    gen = training_generator(graph)
    model.fit_generator(gen, steps_per_epoch=steps_per_epoch_body, epochs=epochs_body, callbacks=[model_checkpoint])

# Read arguments from python command
backbone_name = None
for param in sys.argv:
    if param in ["resnet50", "densenet169"]:
        backbone_name = param

if (backbone_name == None):
    print("Usage: python train_hydra_body.py [resnet50|densenet169]")
    exit(-1)

train_hydra_body(backbone_name)
