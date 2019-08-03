import datetime
import pandas as pd

from pneumothorax_segmentation.tracking.constants import segmentation_columns, segmentation_file_path, classification_columns, classification_file_path

def clear_segmentation_data():
    "Clear data/segmentation.csv"
    df = pd.DataFrame({}, columns=segmentation_columns)
    df.to_csv(segmentation_file_path, header=True, index=False)

def clear_classification_data():
    "Clear data/classification.csv"
    df = pd.DataFrame({}, columns=classification_columns)
    df.to_csv(classification_file_path, header=True, index=False)

clear_segmentation_data()
clear_classification_data()
