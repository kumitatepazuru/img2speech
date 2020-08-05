# define the larger model
from tensorflow.keras.layers import MaxPooling2D, Input, Dense, Dropout, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.models import Model


def create_model(optimizer):
    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="SAME", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(128, (3, 3), padding="SAME", activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="SAME", activation="relu")(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.25)(x)
    y = Dense(62, activation="softmax")(x)

    model = Model(inputs, y)
    print(model.summary())
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model
