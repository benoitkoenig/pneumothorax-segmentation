import tensorflow as tf

from pneumothorax_segmentation.export_predictions.to_csv import clear_outputs_csv, save_line_to_outputs_csv
from pneumothorax_segmentation.preprocess import get_all_images_list
from pneumothorax_segmentation.postprocess import apply_threshold_to_preds, export_mask_to_kaggle_format
from pneumothorax_segmentation.segmentation.predict import get_prediction

def export_predictions():
    "Makes predictions for the test folder and saves them as csv"
    clear_outputs_csv()
    images_list = get_all_images_list("test")

    for (filepath, filename) in images_list:
        prediction = get_prediction(filepath)
        mask = apply_threshold_to_preds(prediction)
        encoded_pixels = export_mask_to_kaggle_format(mask)
        save_line_to_outputs_csv(filename, encoded_pixels)

export_predictions()
