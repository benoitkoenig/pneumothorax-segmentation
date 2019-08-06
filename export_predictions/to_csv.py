import pandas as pd

from pneumothorax_segmentation.constants import folder_path

export_file_path = folder_path + "/results.csv"
columns = ["ImageId", "EncodedPixels"]

def clear_outputs_csv():
    "Resets the file results.csv: clear all the lines if the file exists, create the file otherwise"
    df = pd.DataFrame({}, columns=columns)
    df.to_csv(export_file_path, header=True, index=False)

def save_line_to_outputs_csv(image_id, encoded_pixels):
    "Append a line to results.csv with the given image_id and encoded_pixels"
    df = pd.DataFrame({
        "ImageId": [image_id],
        "EncodedPixels": [encoded_pixels],
    }, columns=columns)
    df.to_csv(export_file_path, mode="a", header=False, index=False)
