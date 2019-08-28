from keras.models import load_model

from pneumothorax_segmentation.constants import folder_path
from pneumothorax_segmentation.preprocess import get_dicom_data, format_pixel_array_for_tf

all_models = []
for body_name in ["resnet50"]: # "densenet169" not included because I did not have the time to train it
    filepath = folder_path + "/weights/hydra_%s_body.hdf5" % body_name
    model = load_model(filepath, compile=False)
    all_models.append(model)

# TODO: loading head models fails due to a bug. To fix

# all_models = []
# for body_name in ["resnet50", "densenet169"]:
#     for head_name in ["none", "resize", "flip_rotate", "filter"]:
#         filepath = folder_path + "/weights/hydra_%s_head_%s.hdf5" % (body_name, head_name)
#         model = load_model(filepath, compile=False)
#         all_models.append(model)

def get_classification_prediction(filepath):
    """
    Returns the prediction as a numpy array of shape (image_size, image_size) for a given filepath\n
    Please note that due a bug loading the models, get_classification_prediction currenly uses the body models.
    Please fix the bug and use the head models of the hydra
    """
    dicom_data = get_dicom_data(filepath)
    image = format_pixel_array_for_tf(dicom_data.pixel_array)
    predictions = [model.predict(image, steps=1) for model in all_models]
    predictions = [p[0][1] for p in predictions]
    return predictions
