import datetime
import pandas as pd

from pneumothorax_segmentation.tracking.constants import columns, file_path

def clear_data():
    "Clear data.csv"
    df = pd.DataFrame({}, columns=columns)
    df.to_csv(file_path, header=True, index=False)

clear_data()
