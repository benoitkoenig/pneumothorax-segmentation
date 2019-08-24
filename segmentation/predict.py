from keras.models import load_model
import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import folder_path, image_size
from pneumothorax_segmentation.preprocess import get_dicom_data, format_pixel_array_for_tf

unet = load_model(folder_path + "/weights/unet.hdf5", compile=False)
sess = tf.Session()

def get_segmentation_prediction(filepath):
    "Returns the prediction as a numpy array of shape (image_size, image_size) for a given filepath"
    dicom_data = get_dicom_data(filepath)
    image = format_pixel_array_for_tf(dicom_data.pixel_array)
    predicted_logits = unet.predict(image, steps=1)
    predictions = tf.convert_to_tensor(predicted_logits, dtype=tf.float32)
    predictions = tf.image.resize(predicted_logits, (image_size, image_size), align_corners=True)
    predictions = sess.run(predictions)
    predictions = np.reshape(predictions, (image_size, image_size))
    return predictions
