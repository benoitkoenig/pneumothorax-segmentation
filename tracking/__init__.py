import datetime
import numpy as np
import pandas as pd

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.tracking.constants import columns, file_path
from pneumothorax_segmentation.postprocess import build_predicted_mask

def calculate_IoU(predicted_mask, true_mask):
    "Calculates the IoU for tracking. This method will most likely be used somewhere else, so it has to be moved when it happens"
    intersection = np.sum(predicted_mask & true_mask)
    union = np.sum(predicted_mask | true_mask)

    if union == 0:
        return 0.

    return intersection / union

def save_data(index, predicted_logits, true_mask):
    "Saves IoUs of images with pneumothorax and wrong diagnosis area of images without. Predicted_mask must be a numpy matrix of shape (image_size, image_size). predicted_logits must be the direct output from the unet model"
    predicted_mask = build_predicted_mask(predicted_logits)
    if (np.max(true_mask) == 0): # No mask on this image
        IoU = None
        wrong_diagnosis = np.sum(predicted_mask)
    else:
        IoU = calculate_IoU(predicted_mask, true_mask)
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
