import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Conv2D, Dropout, MaxPool2D, UpSampling2D

class Unet(Model):
    "Unet model"
    def __init__(self):
        super(Unet, self).__init__()

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
        self.drop5 = Dropout(.5)

        self.up6 = UpSampling2D((2, 2))
        self.concat6 = Concatenate(axis=3)
        self.conv6a = Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv6b = Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.up7 = UpSampling2D((2, 2))
        self.concat7 = Concatenate(axis=3)
        self.conv7a = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv7b = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.up8 = UpSampling2D((2, 2))
        self.concat8 = Concatenate(axis=3)
        self.conv8a = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv8b = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.up9 = UpSampling2D((2, 2))
        self.concat9 = Concatenate(axis=3)
        self.conv9a = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")
        self.conv9b = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer="he_normal")

        self.conv_logits = Conv2D(2, 1, activation="linear", padding="same")

    def call(self, input):
        x1 = self.conv1a(input)
        x1 = self.conv1b(x1)

        x2 = self.pool2(x1)
        x2 = self.conv2a(x2)
        x2 = self.conv2b(x2)

        x3 = self.pool3(x2)
        x3 = self.conv3a(x3)
        x3 = self.conv3b(x3)

        x4 = self.pool4(x3)
        x4 = self.conv4a(x4)
        x4 = self.conv4b(x4)

        x5 = self.pool5(x4)
        x5 = self.conv5a(x5)
        x5 = self.conv5b(x5)
        x5 = self.drop5(x5)

        x6 = self.up6(x5)
        x6 = self.concat6([x4, x6])
        x6 = self.conv6a(x6)
        x6 = self.conv6b(x6)

        x7 = self.up7(x6)
        x7 = self.concat7([x3, x7])
        x7 = self.conv7a(x7)
        x7 = self.conv7b(x7)

        x8 = self.up8(x7)
        x8 = self.concat8([x2, x8])
        x8 = self.conv8a(x8)
        x8 = self.conv8b(x8)

        x9 = self.up9(x8)
        x9 = self.concat9([x1, x9])
        x9 = self.conv9a(x9)
        x9 = self.conv9b(x9)

        logits = self.conv_logits(x9)

        return logits
