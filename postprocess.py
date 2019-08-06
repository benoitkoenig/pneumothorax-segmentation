import numpy as np
import tensorflow as tf

from pneumothorax_segmentation.constants import image_size

def build_predicted_mask(predicted_logits):
    "Inputs the predicted logits as a list of shape (1, tf_image_size, tf_image_size, 2). Outputs a np matrix of shape (image_size, image_size)"
    predictions = tf.convert_to_tensor(predicted_logits, dtype=tf.float32)
    predictions = tf.image.resize(predicted_logits, (image_size, image_size))
    predictions = tf.Session().run(predictions)
    predictions = np.apply_along_axis(lambda l: np.argmax(l), axis=3, arr=predictions)
    predictions = np.reshape(predictions, (image_size, image_size))
    return predictions

def export_mask_to_kaggle_format(input_mask):
    """
        Inputs is the mask as a numpy array. Outputs the mask ready for submission\n
        Note: The predicted logits returned by the model can be converted to the input format via the build_predicted_mask method
    """
    mask = np.transpose(input_mask, (1, 0))
    mask = np.reshape(mask, (-1))
    mask = mask.tolist()
    current_chunk_length = 0
    current_value = 0
    output = []
    while(len(mask) != 0):
        val = mask.pop(0)
        if (val != current_value):
            output.append(current_chunk_length)
            current_chunk_length = 1
            current_value = val
        else:
            current_chunk_length += 1

    if (current_value == 1):
        output.append(current_chunk_length)

    if (len(output) == 0):
        output = "-1"
    else:
        output = [str(o) for o in output]
        output = ", ".join(output)
    return output
