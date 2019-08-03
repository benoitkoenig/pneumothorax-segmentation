segmentation_columns = ["datetime", "index", "IoU", "prediction_area"]
segmentation_file_path = "/".join(__file__.split("/")[:-1]) + "/data/segmentation.csv"

classification_columns = ["datetime", "index", "is_there_pneumothorax", "probs"]
classification_file_path = "/".join(__file__.split("/")[:-1]) + "/data/classification.csv"