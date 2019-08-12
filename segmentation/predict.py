from keras.models import load_model, save_model
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path, tf_image_size

unet = load_model(folder_path + "/weights/unet.hdf5", compile=False)

def get_prediction(image):
    return unet.predict(image, steps=1)
