import datetime
import pandas as pd

from pneumothorax_segmentation.tracking.constants import columns, segmentation_file_path

def clear_segmentation_data():
    "Clear data.csv"
    df = pd.DataFrame({}, columns=columns)
    df.to_csv(segmentation_file_path, header=True, index=False)

clear_segmentation_data()
