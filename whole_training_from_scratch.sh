python -m pneumothorax_segmentation.segmentation.train

python -m pneumothorax_segmentation.hydra_classifier.train_hydra_body densenet169
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_body resnet50

python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head densenet169 none
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head densenet169 resize
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head densenet169 flip_rotate
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head densenet169 filter

python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head resnet50 none
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head resnet50 resize
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head resnet50 flip_rotate
python -m pneumothorax_segmentation.hydra_classifier.train_hydra_head resnet50 filter