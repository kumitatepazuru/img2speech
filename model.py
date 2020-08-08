# define the larger model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, MaxPool2D, Conv2D, Flatten
from tensorflow.keras.models import Sequential


def create_model(optimizer):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(62, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
