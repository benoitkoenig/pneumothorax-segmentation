import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_image_label
from pneumothorax_segmentation.postprocess import interpret_ensemble_classification_predictions
from pneumothorax_segmentation.hydra_classifier.predict import get_classification_prediction

thresholds = [.3, .35, .4]

def get_ratio(input, reverse=False):
    "Input: a list of zeros and ones. Outputs the ratio as a percentage. reverse=True will return the percentage for 1 - ratio"
    mean = np.mean(input)
    if np.isnan(mean):
        return "-"
    if reverse:
        mean = 1 - mean
    return int(100 * mean)

def evaluate():
    images_list = get_all_images_list("train")
    len_images_list = len(images_list)
    guesses = np.empty((len(thresholds), 2, 0)).tolist()
    for (fileindex, (filepath, filename)) in enumerate(images_list):
        label = get_image_label(filename)
        predictions = get_classification_prediction(filepath)
        for i in range(len(thresholds)):
            predicted_label = interpret_ensemble_classification_predictions(predictions, thresholds[i])
            guesses[i][label].append(predicted_label)

        print("\nThreshold\tLabel 0\t\tLabel 1\t\t%s/%s" % (fileindex, len_images_list))
        for i in range(len(thresholds)):
            print("%s:\t\t%s%%\t\t%s%%" % (thresholds[i], get_ratio(guesses[i][0], reverse=True), get_ratio(guesses[i][1])))

evaluate()
