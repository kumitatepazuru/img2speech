# define the larger model
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, \
    Activation, Flatten
from keras.models import Sequential


def create_model():
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), padding="same", input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(62))
    model.add(Activation('softmax'))

    print(model.summary())
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model
