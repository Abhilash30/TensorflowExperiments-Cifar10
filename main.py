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
from keras.models import Sequential
from keras.utils import to_categorical

K.set_image_data_format("channels_last")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_train.shape[1]  # 10 classes


def baseline_model():
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(32, 32, 3),
            ),
            Conv2D(32, (3, 3), padding="same", activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
                                        #too much pooling loses information, so we can reduce the number of pooling layers
            Flatten(),
            # Assuming your previous layer has a lot of inputs (like an image)
            Dense(128, activation="relu"),
            Dropout(0.3),  # Add dropout for regularization
            # Dense(512, activation="relu"),   Too many layers can lead to overfitting, so we can reduce the number of dense layers
            # Dropout(0.5),  # Add dropout for regularization
            # Dense(128, activation="relu"),
            # Dropout(0.5),  # Add dropout for regularization
            # Dense(128, activation="relu"),
            # Dropout(0.5),  # Add dropout for regularization

            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = baseline_model()
    # model.summary() works better BEFORE fit so you can verify the structure
    model.summary()


    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    History = model.fit(x_train, y_train, epochs=18, validation_data=(x_test, y_test), callbacks=[early_stopping, checkpoint])

    

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

    plt.savefig("training_history.png")
    plt.close()
