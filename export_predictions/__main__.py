import tensorflow as tf

from pneumothorax_segmentation.export_predictions.to_csv import clear_outputs_csv, save_line_to_outputs_csv
from pneumothorax_segmentation.hydra_classifier.predict import get_classification_prediction
from pneumothorax_segmentation.params import classification_threshold, segmentation_threshold
from pneumothorax_segmentation.preprocess import get_all_images_list
from pneumothorax_segmentation.postprocess import apply_threshold_to_preds, interpret_ensemble_classification_predictions, export_mask_to_kaggle_format
from pneumothorax_segmentation.segmentation.predict import get_segmentation_prediction

def export_predictions():
    "Makes predictions for the test folder and saves them as csv"
    clear_outputs_csv()
    images_list = get_all_images_list("stage-2")

    for (filepath, filename) in images_list:
        classification_predictions = get_classification_prediction(filepath)
        predicted_label = interpret_ensemble_classification_predictions(classification_predictions, classification_threshold)
        if (predicted_label == 0):
            encoded_pixels = "-1"
        else:
            prediction = get_segmentation_prediction(filepath)
            mask = apply_threshold_to_preds(prediction, segmentation_threshold)
            encoded_pixels = export_mask_to_kaggle_format(mask)
        save_line_to_outputs_csv(filename, encoded_pixels)

export_predictions()
