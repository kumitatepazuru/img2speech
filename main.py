from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

from model import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# from keras import backend as K

# K.set_image_data_format('channels_first')

x_train, y_train = extract_training_samples("byclass")
x_test, y_test = extract_test_samples('byclass')

# Dataset images preprocessing
x_train = x_train.reshape((-1, x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((-1, x_train.shape[1], x_train.shape[2], 1))
# x_train, x_test = np.transpose(x_train, [0, 3, 1, 2]), np.transpose(x_test, [0, 3, 1, 2])

# Dataset labels preprocessing
y_train = to_categorical(y_train, 62)
y_test = to_categorical(y_test, 62)

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=[0.75, 1.25],
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(x_train)

# Create a model
optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
model = create_model(optimizer)

# Train
batch_size = 128

history = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    epochs=60,
    callbacks=[
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=0.0001,
            verbose=1
        ),
        ModelCheckpoint(
            "model.{epoch:02d}-{accuracy:.4f}.hdf5",
            monitor="accuracy",
            verbose=1
        )
    ],
    validation_data=(x_test, y_test)
)

model.save("model.h5")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))

# Show predict images
for i in range(10):
    plt.imshow(x_test[i].reshape((28, 28)), 'gray')
plt.show()

# Show predict labels
test_predictions = model.predict(x_test[0:10])
test_predictions = np.argmax(test_predictions, axis=1)
print(test_predictions)
