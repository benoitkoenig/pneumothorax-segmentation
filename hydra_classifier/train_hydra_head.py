import keras.backend as K
from keras.models import load_model, save_model
import sys
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.hydra_classifier.params import steps_per_epoch, epochs, head_iterations
from pneumothorax_segmentation.hydra_classifier.training_generator import get_training_generator

# Hydra: an Ensemble of Convolutional NeuralNetworks for Geospatial Land Classification: https://arxiv.org/pdf/1802.03518.pdf

graph = tf.compat.v1.get_default_graph()

def train_hydra_head(backbone_name, data_augment_technique):
    filepath = folder_path + "/weights/hydra_%s_head_%s.hdf5" % (backbone_name, data_augment_technique)
    filepath_body = folder_path + "/weights/hydra_%s_body.hdf5" % backbone_name
    for iteration in range(head_iterations):
        training_generator = get_training_generator()
        for epoch_index in range(epochs): # See note in segmentation/training
            graph = tf.Graph()
            K.clear_session()
            gen = training_generator(graph, starting_index=epoch_index * steps_per_epoch)
            with graph.as_default():
                if (iteration == 0 & epoch_index == 0):
                    model = load_model(filepath_body)
                else:
                    model = load_model(filepath)
                model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=1)
                save_model(model, filepath)

# Read arguments from python command
backbone_name = None
for param in sys.argv:
    if param in ["resnet50", "densenet169"]:
        backbone_name = param

data_augment_technique = None
for param in sys.argv:
    if param in ["none", "resize", "flip_rotate", "filter"]:
        data_augment_technique = param

if (backbone_name == None) | (data_augment_technique == None):
    print("Usage: python train_hydra_head.py [resnet50|densenet169] [none|resize|flip_rotate|filter]")
    exit(-1)

train_hydra_head(backbone_name, data_augment_technique)
