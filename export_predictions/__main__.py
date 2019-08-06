import tensorflow as tf

from pneumothorax_segmentation.export_predictions.testing_generator import testing_generator
from pneumothorax_segmentation.export_predictions.to_csv import clear_outputs_csv, save_line_to_outputs_csv
from pneumothorax_segmentation.postprocess import build_predicted_mask, export_mask_to_kaggle_format
from pneumothorax_segmentation.segmentation.predict import get_prediction

graph = tf.compat.v1.get_default_graph()

def export_predictions():
    "Makes predictions for the test folder and saves them as csv"
    clear_outputs_csv()
    for (image, filename) in testing_generator(graph):
        predicted_logits = get_prediction(image)
        mask = build_predicted_mask(predicted_logits)
        encoded_pixels = export_mask_to_kaggle_format(mask)
        save_line_to_outputs_csv(filename, encoded_pixels)

export_predictions()
