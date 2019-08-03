import datetime
import numpy as np
import pandas as pd

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.tracking.constants import segmentation_columns, segmentation_file_path, classification_columns, classification_file_path
from pneumothorax_segmentation.postprocess import build_predicted_mask

def calculate_IoU(predicted_mask, true_mask):
    "Calculates the IoU for tracking. This method will most likely be used somewhere else, so it has to be moved when it happens"
    intersection = np.sum(predicted_mask & true_mask)
    union = np.sum(predicted_mask | true_mask)

    if union == 0:
        return 0.

    return intersection / union

def save_segmentation_data(index, predicted_logits, true_mask):
    "Saves IoUs of images with pneumothorax and wrong diagnosis area of images without. Predicted_mask must be a numpy matrix of shape (image_size, image_size). predicted_logits must be the direct output from the unet model"
    predicted_mask = build_predicted_mask(predicted_logits)
    IoU = None
    if (np.max(true_mask) == 1): # If this image has a ground truth mask
        IoU = calculate_IoU(predicted_mask, true_mask)
    prediction_area = np.sum(predicted_mask)

    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "index": [index],
        "IoU": [IoU],
        "prediction_area": [prediction_area],
    }, columns=segmentation_columns)
    df.to_csv(segmentation_file_path, mode="a", header=False, index=False)

def save_classification_data(index, is_there_pneumothorax, probs):
    "Saves probabilities for predicting pneumothorax in a picture"
    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "index": [index],
        "is_there_pneumothorax": [is_there_pneumothorax],
        "probs": [probs],
    }, columns=classification_columns)
    df.to_csv(classification_file_path, mode="a", header=False, index=False)

def get_segmentation_dataframes():
    return pd.read_csv(segmentation_file_path)
