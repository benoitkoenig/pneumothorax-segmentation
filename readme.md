## Pneumothorax Segmentation

This repository is my solution to this [Kaggle competition](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

Author: Benoît Koenig

### Usage:

- Training the segmentation model: python -m segmentation.train

- Training the hydra classifier's body: python -m hydra_classifier.train_hydra_body [resnet50|densenet169]

- Training the hydra classifier's heads: python -m hydra_classifier.train_hydra_head [resnet50|densenet169] [none|resize|flip_rotate|filter]

- Evaluating the current model (shows the dice coefficient from the training data): python -m evaluate

- Exporting final predictions to results.csv: python -m export_predictions

- Check data augmentation: python -m visualization.show_data_augment [index]

- Visualize prediction: python -m visualization.show_prediction [train|test] [index]

- Show raw data from dicom: python -m visualization.show_raw_data [train|test] [index]

- Show ground truth mask: python -m visualization.show_true_mask [index]

#### Note

tensorflow-gpu is not included in requirements.txt as it is not always relevant. To use the GPU, install tensorflow-gpu via "pip install tensorflow-gpu"
