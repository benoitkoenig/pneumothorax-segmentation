import numpy as np
import tensorflow as tf
import unittest

from pneumothorax_segmentation.segmentation.params import mask_factor, non_mask_factor
from pneumothorax_segmentation.segmentation.loss import get_bce_loss

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

class TestBceLoss(unittest.TestCase):
    def test_with_mask_and_good_results(self):
        true_mask = tf.convert_to_tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=tf.float32)
        true_mask = tf.reshape(true_mask, (1, 4, 4, 1))
        probs = tf.convert_to_tensor([[0.2, 0.2, 0.2, 0.2], [0.2, 0.9, 0.9, 0.9], [0.2, 0.9, 0.9, 0.2], [0.2, 0.2, 0.2, 0.2]], dtype=tf.float32)
        probs = tf.reshape(probs, (1, 4, 4, 1))
        loss = get_bce_loss(true_mask, probs)
        self.assertTrue(loss.numpy() < 1)

    def test_with_mask_and_bad_results(self):
        true_mask = tf.convert_to_tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=tf.float32)
        true_mask = tf.reshape(true_mask, (1, 4, 4, 1))
        probs = tf.convert_to_tensor([[0.8, 0.8, 0.8, 0.8], [0.8, 0.2, 0.2, 0.2], [0.8, 0.2, 0.2, 0.8], [0.8, 0.8, 0.8, 0.8]], dtype=tf.float32)
        probs = tf.reshape(probs, (1, 4, 4, 1))
        loss = get_bce_loss(true_mask, probs)
        self.assertTrue(loss.numpy() > 1)

    def test_without_mask(self):
        true_mask = tf.convert_to_tensor(np.zeros((1, 4, 4, 1)), dtype=tf.float32)
        probs = tf.convert_to_tensor(np.zeros((1, 4, 4, 1)) + .2, dtype=tf.float32)
        loss = get_bce_loss(true_mask, probs)
        self.assertTrue(loss.numpy() < 1)

    def test_proportions_respected(self):
        true_mask_only_ones = tf.convert_to_tensor(np.ones((1, 4, 4, 1)), dtype=tf.float32)
        probs_only_ones = tf.convert_to_tensor(np.zeros((1, 4, 4, 1)) + 0.7, dtype=tf.float32)
        loss_only_ones = get_bce_loss(true_mask_only_ones, probs_only_ones)

        true_mask_only_zeros = tf.convert_to_tensor(np.zeros((1, 4, 4, 1)), dtype=tf.float32)
        probs_only_zeros = tf.convert_to_tensor(np.zeros((1, 4, 4, 1)) + 0.2, dtype=tf.float32)
        loss_only_zeros = get_bce_loss(true_mask_only_zeros, probs_only_zeros)

        true_mask = tf.convert_to_tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=tf.float32)
        true_mask = tf.reshape(true_mask, (1, 4, 4, 1))
        probs = tf.convert_to_tensor([[0.2, 0.2, 0.2, 0.2], [0.2, 0.7, 0.7, 0.7], [0.2, 0.7, 0.7, 0.2], [0.2, 0.2, 0.2, 0.2]], dtype=tf.float32)
        probs = tf.reshape(probs, (1, 4, 4, 1))
        loss = get_bce_loss(true_mask, probs)
        self.assertEqual(loss.numpy(), (loss_only_ones + loss_only_zeros).numpy())

if __name__ == "__main__":
    unittest.main()
