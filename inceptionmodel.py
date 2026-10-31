import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
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
from keras.models import Model, Sequential
from keras.utils import to_categorical

K.set_image_data_format("channels_last")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_train.shape[1]  # 10 classes

from keras.layers import Input, concatenate

def inception_module(x, f1, f3, f5, fp):
    # Branch 1: 1x1 reduces channels, then 3x3
    t1 = Conv2D(f1, (1, 1), padding="same", activation="relu")(x)
    t1 = Conv2D(f3, (3, 3), padding="same", activation="relu")(t1)
    # Branch 2: 1x1 reduces channels, then 5x5
    t2 = Conv2D(f1, (1, 1), padding="same", activation="relu")(x)
    t2 = Conv2D(f5, (5, 5), padding="same", activation="relu")(t2)
    # Branch 3: pooling, then 1x1
    t3 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    t3 = Conv2D(fp, (1, 1), padding="same", activation="relu")(t3)
    return concatenate([t1, t2, t3], axis=3)
def baseline_inception_model():
    inp = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = inception_module(x, f1=32, f3=64, f5=32, fp=32)  # out: 16x16x128
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
if __name__ == "__main__":
    model = baseline_inception_model()
    model.summary()
    History = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

    

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
