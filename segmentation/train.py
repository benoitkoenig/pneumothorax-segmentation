from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, save_model
from keras.optimizers import Adam
from segmentation_models import Unet
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path, image_size, tf_image_size
from pneumothorax_segmentation.segmentation.loss import calculate_loss
from pneumothorax_segmentation.segmentation.params import learning_rate, steps_per_epoch, epochs
from pneumothorax_segmentation.segmentation.training_generator import training_generator

file_path = folder_path + "/weights/unet.hdf5"

def train():
    for epoch_index in range(epochs):
        # Why iterating over epochs instead of passing them correctly as a parameter of fit_generator?
        # The problem is that I need to clean the graph regularly, or I will get a memory error
        # And I did not find a way to clean the memory from used operations without resetting the whole graph
        # But resetting the whole graph also requires to re-load the wholes weights
        # This is obviously not an acceptable long-term solution. To see the issue on github:
        # https://github.com/tensorflow/tensorflow/issues/31419
        graph = tf.Graph()
        K.clear_session()
        gen = training_generator(graph, starting_index=epoch_index * steps_per_epoch)
        with graph.as_default():
            if (epoch_index == 0):
                unet = Unet(
                    "resnet34",
                    encoder_weights="imagenet",
                    classes=1,
                    activation="sigmoid",
                    input_shape=(tf_image_size, tf_image_size, 3),
                )
                unet.compile(optimizer=Adam(lr=learning_rate), loss=calculate_loss)
            else:
                unet = load_model(file_path, custom_objects={"calculate_loss": calculate_loss})
            unet.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=1)
            save_model(unet, file_path)

train()
