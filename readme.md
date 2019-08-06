## Pneumothorax Segmentation

This repository is my solution to this [Kaggle competition](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

Author: Benoît Koenig

### Usage:

- Training the model: python -m segmentation.train

- Exporting final predictions to results.csv: python -m export_predictions

- Check data augmentation: python -m visualization.show_data_augment [index]

- Visualize prediction: python -m visualization.show_prediction [train|test] [index]

- Show raw data from dicom: python -m visualization.show_raw_data [train|test] [index]

- Show ground truth mask: python -m visualization.show_true_mask [index]

#### Note

tensorflow-gpu is not included in requirements.txt as it is not always relevant. To use the GPU, install tensorflow-gpu via "pip install tensorflow-gpu"
