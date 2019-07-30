import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

class Classifier(Model):
    "Classifier model"
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1a = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv1b = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.pool2 = MaxPool2D((2, 2))
        self.conv2a = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv2b = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.pool3 = MaxPool2D((2, 2))
        self.conv3a = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv3b = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.pool4 = MaxPool2D((2, 2))
        self.conv4a = Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv4b = Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.pool5 = MaxPool2D((2, 2))
        self.conv5a = Conv2D(1024, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv5b = Conv2D(1024, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.flat6 = Flatten()
        self.dense6 = Dense(2048, activation="relu")

        self.dense_logits = Dense(2, activation="linear")

    def call(self, input):
        x = self.conv1a(input)
        x = self.conv1b(x)

        x = self.pool2(x)
        x = self.conv2a(x)
        x = self.conv2b(x)

        x = self.pool3(x)
        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.pool4(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        x = self.pool5(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = self.flat6(x)
        x = self.dense6(x)

        logits = self.dense_logits(x)

        return logits
