from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from segmentation_models import Unet
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path, image_size, tf_image_size
from pneumothorax_segmentation.segmentation.loss import get_bce_loss
from pneumothorax_segmentation.segmentation.params import learning_rate, steps_per_epoch, epochs
from pneumothorax_segmentation.segmentation.training_generator import training_generator, generator_length

graph = tf.compat.v1.get_default_graph()

if (generator_length != steps_per_epoch * epochs):
    print("\n Warning: The generator's length is different from steps_per_epoch * epochs: %s != %s * %s" % (generator_length, steps_per_epoch, epochs))

def train():
    unet = Unet(
        'resnet34',
        encoder_weights='imagenet',
        classes=1,
        activation="sigmoid",
        input_shape=(tf_image_size, tf_image_size, 3),
    )
    unet.compile(optimizer=Adam(lr=learning_rate), loss=get_bce_loss)

    model_checkpoint = ModelCheckpoint(folder_path + "/weights/unet.hdf5")

    gen = training_generator(graph)
    unet.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[model_checkpoint])

train()
