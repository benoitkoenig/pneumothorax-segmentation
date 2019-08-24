import numpy as np

from pneumothorax_segmentation.params import pred_threshold

def apply_threshold_to_preds(preds, threshold=pred_threshold):
    """
    Applies a given threshold to preds, returning a numpy array of same shape as preds,
    containing 0 or 1 depending if the value is greater than the threshold\n
    If threshold is unspecified, the value from params/pred_threshold is used
    """
    return (preds > threshold).astype(int)

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
    for val in mask:
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
        output = " ".join(output)
    return output
