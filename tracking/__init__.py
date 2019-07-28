import datetime
import numpy as np
import pandas as pd

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.tracking.constants import columns, file_path

def calculate_IoU(predicted_logits, labels):
    "Calculates the IoU for tracking. This method will most likely be used somewhere else, so it has to be moved when it happens"
    predictions = np.apply_along_axis(lambda l: np.argmax(l), axis=3, arr=predicted_logits)

    intersection = np.sum(labels & predictions)
    union = np.sum(labels | predictions)

    if union == 0:
        return 0.

    return intersection / union

def calculate_predicted_area(predicted_logits):
    "Calculates the area where the pneumothorax was predicted"
    predictions = np.apply_along_axis(lambda l: np.argmax(l), axis=3, arr=predicted_logits)

    return np.sum(predictions)

def save_data(index, predicted_logits, labels):
    "Saves IoUs of images with pneumothorax and wrong diagnosis area of images without. Predicted_logits must be a np matrix of shape (1, n, n, 2) and labels a np matrix of shape (1, n, n)"
    if (np.max(labels) == 0): # No mask on this image
        IoU = None
        wrong_diagnosis = calculate_predicted_area(predicted_logits)
    else:
        IoU = calculate_IoU(predicted_logits, labels)
        wrong_diagnosis = None

    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "index": [index],
        "IoU": [IoU],
        "wrong_diagnosis": [wrong_diagnosis],
    }, columns=columns)
    df.to_csv(file_path, mode="a", header=False, index=False)

# def get_dataframes():
#     return pd.read_csv(file_path)
