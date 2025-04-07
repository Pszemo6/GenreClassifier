import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "data.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # konwersja donumpy arrays
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    #print("Data succesfully loaded!")

    return  x, y

def plot_history(history):

    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy", color="blue")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy", color="red")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")

    axs[1].plot(history.history["loss"], label="train error", color="blue")
    axs[1].plot(history.history["val_loss"], label="test error", color="red")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")

    plt.show()

if __name__ == "__main__":

    x, y = load_data(DATA_PATH)

    # podział danych na treningowe i testowe
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # tworzymy model MLP
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),

        # 1st dense layer
        #keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        #keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        #keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # kompilacja modelu
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # trening
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

    plot_history(history)