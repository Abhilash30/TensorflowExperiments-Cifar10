import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions 
from keras import backend as K
from tensorflow.keras import layers, models, Input 
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt 
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import BatchNormalization, Activation

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
    RandomFlip,
    RandomTranslation,
)
K.set_image_data_format("channels_last")


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_train.shape[1] 

def vgg16_model():
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    for filters, n_convs in [(64, 2), (128, 2), (256, 2), (512, 2)]:
        for _ in range(n_convs):
            model.add(Conv2D(filters, (3, 3), padding="same", use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation="relu")) 

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_vgg16.keras", monitor="val_loss", save_best_only=True)
    model = vgg16_model()
    model.summary()
    History = model.fit(x_train, y_train, epochs=90, validation_data=(x_test, y_test), callbacks=[early_stop, checkpoint])

    

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(History.history["accuracy"])
    plt.plot(History.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(History.history["loss"])
    plt.plot(History.history["val_loss"])   
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    plt.savefig("training_history1.png")
    plt.close()
