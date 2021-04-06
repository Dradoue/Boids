from keras.layers import Input, Conv1D, Dropout, Conv1DTranspose, Dense, MaxPooling1D, Flatten, BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import keras
from tensorflow.python.keras import Sequential

from utils import get_positions_velocity_headings

from sklearn.metrics import confusion_matrix


def create_dataset(repository="simulation_data_30-30-30-30/", filename="",
                   step_init=300, step_end=6500):
    # build labels
    labels = np.array([0] + [1] + [2] + [3])[:, np.newaxis]
    enc = OneHotEncoder()
    enc.fit(X=labels)
    labels_encoded = enc.transform(labels).toarray()

    use_headings = True

    # build dataset of T * n * (positions + headings)
    data = list()
    labels = list()

    for t in range(step_init, step_end, 1):
        positions, velocities, headings = \
            get_positions_velocity_headings(repository, filename, t)

        # normalise data
        positions = positions / 2000
        headings = headings / np.pi

        # add new axis to headings for dimension match
        headings_1 = headings[0:30, np.newaxis]
        headings_2 = headings[30:60, np.newaxis]
        headings_3 = headings[60:90, np.newaxis]
        headings_4 = headings[90:120, np.newaxis]

        positions_1 = positions[0:30, :]
        positions_2 = positions[30:60, :]
        positions_3 = positions[60:90, :]
        positions_4 = positions[90:120, :]

        if use_headings:
            # concatenate positions and headings
            data_1 = np.concatenate((positions_1, headings_1), axis=1).reshape(1, 90)
            data_2 = np.concatenate((positions_2, headings_2), axis=1).reshape(1, 90)
            data_3 = np.concatenate((positions_3, headings_3), axis=1).reshape(1, 90)
            data_4 = np.concatenate((positions_4, headings_4), axis=1).reshape(1, 90)

            data.append(data_1)
            data.append(data_2)
            data.append(data_3)
            data.append(data_4)

            labels.append(labels_encoded[0, :])
            labels.append(labels_encoded[1, :])
            labels.append(labels_encoded[2, :])
            labels.append(labels_encoded[3, :])

        else:

            data.append(positions_1.reshape(1, 60))
            data.append(positions_1.reshape(1, 60))
            data.append(positions_1.reshape(1, 60))
            data.append(positions_1.reshape(1, 60))

            labels.append(labels_encoded[0, :])
            labels.append(labels_encoded[1, :])
            labels.append(labels_encoded[2, :])
            labels.append(labels_encoded[3, :])

    data_ = np.array(data, dtype=float)
    labels_ = np.array(labels, dtype=int)
    print("data and labels built")
    print("data shape ", data_.shape)
    print("labels shape ", labels_.shape)
    print(data_)
    print(labels_)

    if use_headings:
        np.savetxt("data/data.txt", data_.reshape(((6500 - 300) * 4, 90)))
        np.savetxt("data/targets.txt", labels_.reshape(((6500 - 300) * 4, 4)))
    else:
        np.savetxt("data/data.txt", data_.reshape(((6500 - 300) * 4, 60)))
        np.savetxt("data/targets.txt", labels_.reshape(((6500 - 300) * 4, 4)))

    return data_, labels_


def build_conv1D_network():
    model = Sequential()
    model.add(Conv1D(4, 25, activation="relu", input_shape=(90, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(4, 25, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


def train_network(model, data, targets, model_name="model"):

    choice_split = 1

    print(data)
    print(data.shape)

    if choice_split == 0:

        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.4, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=1)

    else:

        X_train, X_test, X_val, y_train, y_test, y_val = data[:15000, :], data[15000:20000, :], data[20000:, :], \
                                                         targets[:15000, :], targets[15000:20000, :], targets[20000:, :]

    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=50,
        verbose=1,
        shuffle=True,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_val, y_val),
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict_classes(X_test)

    confmat = confusion_matrix(y_pred, np.argmax(y_test, axis=1))

    df_cm = pd.DataFrame(confmat)
    print(confmat)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights("models/" + "model_name" + ".h5")

    return model


def experiment_network():
    data, targets = create_dataset(repository="simulation_data/")
    # data = np.loadtxt("data.txt", dtype=float)
    # targets = np.loadtxt("targets.txt")
    data = data.reshape((24800, 90, 1))
    model = build_conv1D_network()
    train_network(model, data, targets)


if __name__ == '__main__':
    experiment_network()
