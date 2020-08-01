from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from model import create_model
from keras.preprocessing.image import ImageDataGenerator
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


def step_decay(epoch):
    lr = 0.1
    if epoch % 10 == 0:
        lr /= 2
    return lr


train_images, train_labels = extract_training_samples("byclass")
test_images, test_labels = extract_test_samples('byclass')

# Dataset images preprocessing
train_images = train_images.reshape((-1, train_images.shape[1], train_images.shape[2], 1))
test_images = test_images.reshape((-1, train_images.shape[1], train_images.shape[2], 1))

# Dataset labels preprocessing
train_labels = to_categorical(train_labels, 62)
test_labels = to_categorical(test_labels, 62)

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.1
)

datagen.fit(train_images)

# Create a model
model = create_model()

# Train
batch_size = 128

history = model.fit_generator(
    datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=len(train_images) / batch_size,
    epochs=150,
    workers=12,
    use_multiprocessing=True,
    callbacks=[
        LearningRateScheduler(step_decay),
        ModelCheckpoint("model.{epoch:02d}-{acc:.2f}.hdf5",monitor="acc")
    ]
)

model.save("model.h5")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))

# Show predict images
for i in range(10):
    plt.imshow(test_images[i].reshape((28, 28)), 'gray')
plt.show()

# Show predict labels
test_predictions = model.predict(test_images[0:10])
test_predictions = np.argmax(test_predictions, axis=1)
print(test_predictions)
