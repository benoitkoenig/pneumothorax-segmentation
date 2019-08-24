from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model

from pneumothorax_segmentation.params import tf_image_size

def get_classifier(backbone_name):
    "Returns a classifier model. backbone_name must be one of 'resnet50', 'densenet169'"
    if (backbone_name == "resnet50"):
        backbone = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(tf_image_size, tf_image_size, 3),
        )
    elif (backbone_name == "densenet169"):
        backbone = DenseNet169(
            include_top=False,
            weights="imagenet",
            input_shape=(tf_image_size, tf_image_size, 3),
        )
    else:
        print("get_classifier called with an invalid backbone_name. Exiting immediatly")
        exit(-1)

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(2, activation='softmax')(x)
    classifier = Model(inputs=backbone.input, outputs=predictions)

    return classifier
