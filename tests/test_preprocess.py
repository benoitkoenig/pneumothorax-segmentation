import numpy as np
import tensorflow as tf
import unittest

from pneumothorax_segmentation.constants import image_size
from pneumothorax_segmentation.params import tf_image_size
from pneumothorax_segmentation.preprocess import format_pixel_array_for_tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

class TestFormatPixelArrayForTf(unittest.TestCase):
    def test_with_standard_image(self):
        input_matrix = np.zeros((image_size, image_size), dtype=np.float32)
        input_matrix[0, 0] = 255.
        input_matrix[image_size - 1, 0] = 127.
        output = format_pixel_array_for_tf(input_matrix)
        self.assertEqual(output.shape, [1, tf_image_size, tf_image_size, 3])
        output_value = output.numpy()
        self.assertNotEqual(output_value[0, 0, 0, 0], 0.)
        self.assertNotEqual(output_value[0, tf_image_size - 1, 0, 0], 0.)
        self.assertEqual(output_value[0, 0, tf_image_size - 1, 0], 0.)
        self.assertEqual(output_value[0, tf_image_size - 1, tf_image_size - 1, 0], 0.)

if __name__ == "__main__":
    unittest.main()
