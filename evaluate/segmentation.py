import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.preprocess import get_all_images_list, get_true_mask, get_image_label
from pneumothorax_segmentation.postprocess import apply_threshold_to_preds
from pneumothorax_segmentation.segmentation.predict import get_segmentation_prediction

thresholds = [.1, .5, .9]

def calculate_dice_coefficient(X, Y):
    "X and Y must be numpy arrays of same shape and values 0 or 1. Returns the dice coefficient"
    return 2 * np.sum(X * Y) / np.sum(X + Y) # np.sum(X + Y) = sum(X) + sum(Y)

def evaluate():
    images_list = get_all_images_list("train")
    len_images_list = len(images_list)
    dices_coefficients = [[], [], []]
    wrong_diagnosis_areas = [[], [], []]
    for (fileindex, (filepath, filename)) in enumerate(images_list):
        label = get_image_label(filename)
        true_mask = get_true_mask(filename)
        predictions = get_segmentation_prediction(filepath)
        for i in range(3):
            predicted_mask = apply_threshold_to_preds(predictions, thresholds[i])
            if (label == 1):
                dice_coef = calculate_dice_coefficient(predicted_mask, true_mask)
                dices_coefficients[i].append(dice_coef)
            else:
                area = np.sum(predicted_mask) / (image_size ** 2)
                wrong_diagnosis_areas[i].append(area)

        print("\n%s/%s Avg dice coef and wrong diagnosis area for threshold" % (fileindex, len_images_list))
        for i in range(3):
            print("%s\t%s\t%s" % (thresholds[i], np.mean(dices_coefficients[i]), np.mean(wrong_diagnosis_areas[i])))

evaluate()
