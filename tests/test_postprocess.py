import numpy as np
import unittest

from pneumothorax_segmentation.postprocess import apply_threshold_to_preds, export_mask_to_kaggle_format

class TestPostprocessMethods(unittest.TestCase):
    def test_apply_threshold_to_preds(self):
        input = np.array([0.2, 0.4, 0.6, 0.8])
        output = apply_threshold_to_preds(input, .5)
        self.assertListEqual(output.tolist(), [0, 0, 1, 1])

    def test_apply_threshold_to_preds_with_default_param(self):
        input = np.array([1e-7, 1e-7, 1 - 1e-7, 1 - 1e-7])
        output = apply_threshold_to_preds(input)
        self.assertListEqual(output.tolist(), [0, 0, 1, 1])

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

if __name__ == "__main__":
    unittest.main()
