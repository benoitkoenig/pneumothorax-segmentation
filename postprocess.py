import numpy as np

def apply_threshold_to_preds(preds, threshold):
    "Returns a numpy array of same shape as preds, containing 1 where the value is greater than the threshold, 0 elsewhere"
    return (preds > threshold).astype(int)

def interpret_ensemble_classification_predictions(preds, threshold):
    "Given a list of probabilities, returns 1 if the average is above the given threshold, 0 otherwise"
    return int(np.mean(preds) > threshold)

def export_mask_to_kaggle_format(input_mask):
    "Inputs is the mask as a numpy array. Outputs the mask ready for submission"
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
