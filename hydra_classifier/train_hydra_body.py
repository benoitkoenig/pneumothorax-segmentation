import keras.backend as K
from keras.models import load_model, save_model
from keras.optimizers import Adam
import sys
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.hydra_classifier.get_classifier import get_classifier
from pneumothorax_segmentation.hydra_classifier.params import learning_rate, steps_per_epoch, epochs
from pneumothorax_segmentation.hydra_classifier.training_generator import get_training_generator

# Hydra: an Ensemble of Convolutional NeuralNetworks for Geospatial Land Classification: https://arxiv.org/pdf/1802.03518.pdf

graph = tf.compat.v1.get_default_graph()

def train_hydra_body(backbone_name):
    filepath = folder_path + "/weights/hydra_%s_body.hdf5" % backbone_name
    training_generator = get_training_generator()
    for epoch_index in range(epochs): # See note in segmentation/training
        graph = tf.Graph()
        K.clear_session()
        gen = training_generator(graph, starting_index=epoch_index * steps_per_epoch)
        with graph.as_default():
            if (epoch_index == 0):
                model = get_classifier(backbone_name)
                model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
            else:
                model = load_model(filepath)
            model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=1)
            save_model(model, filepath)

# Read arguments from python command
backbone_name = None
for param in sys.argv:
    if param in ["resnet50", "densenet169"]:
        backbone_name = param

if (backbone_name == None):
    print("Usage: python train_hydra_body.py [resnet50|densenet169]")
    exit(-1)

train_hydra_body(backbone_name)
