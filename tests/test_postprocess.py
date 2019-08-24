import numpy as np
import unittest

from pneumothorax_segmentation.postprocess import apply_threshold_to_preds, interpret_ensemble_classification_predictions, export_mask_to_kaggle_format
from pneumothorax_segmentation.preprocess import get_all_images_list, get_raw_masks, get_true_mask

class TestPostprocessMethods(unittest.TestCase):
    def test_apply_threshold_to_preds(self):
        input = np.array([0.2, 0.4, 0.6, 0.8])
        output = apply_threshold_to_preds(input, .5)
        self.assertListEqual(output.tolist(), [0, 0, 1, 1])

    def test_interpret_ensemble_classification_predictions_with_low_probs(self):
        input = np.array([0.2, 0.4, 0.6])
        output = interpret_ensemble_classification_predictions(input, .5)
        self.assertEqual(output, 0)

    def test_interpret_ensemble_classification_predictions_with_high_probs(self):
        input = np.array([.4, .6, .8])
        output = interpret_ensemble_classification_predictions(input, .5)
        self.assertEqual(output, 1)

    def test_export_mask_to_kaggle_format(self):
        input_matrix = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        output = export_mask_to_kaggle_format(input_matrix)
        self.assertEqual(output, "6 1 2 2 2 2")

    def test_export_mask_to_kaggle_format_with_no_mask(self):
        input_matrix = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        output = export_mask_to_kaggle_format(input_matrix)
        self.assertEqual(output, "-1")

    def test_export_mask_to_kaggle_format_with_border_masks(self):
        input_matrix = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ])
        output = export_mask_to_kaggle_format(input_matrix)
        self.assertEqual(output, "0 3 14 3")

    def test_compose_get_true_mask_with_export_mask_yields_raw_mask(self):
        (_, filename) = get_all_images_list("train")[5] # This picture has exactly one mask
        raw_mask = get_raw_masks(filename)
        true_mask = get_true_mask(filename)
        output = " " + export_mask_to_kaggle_format(true_mask)
        self.assertEqual(output, raw_mask[0])

if __name__ == "__main__":
    unittest.main()
