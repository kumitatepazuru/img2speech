# パッケージのインポート
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

import load_data
from model import larger_model

train_paths = []
for root, dirs, files in tqdm(os.walk("./font")):
    train_paths += list(map(lambda n: root + "/" + n, files))

val_count = int(len(train_paths) * 0.01)

train_gen = load_data.Generator(
    train_paths[val_count:],
    batch_size=4096)
val_gen = load_data.Generator(
    train_paths[:val_count],
    batch_size=4096)

# Building the model
model = larger_model()

# model = Sequential()
# model.add(Dense(1024, activation='relu', input_shape=(32 ** 2,)))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Dense(94, activation='softmax'))  # 出力層

# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 1)))
# model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.5))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(94))
# model.add(Activation('softmax'))

# 学習
history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_gen.num_batches_per_epoch,
    validation_data=val_gen,
    validation_steps=val_gen.num_batches_per_epoch,
    epochs=30,
    use_multiprocessing=True,
    workers=12,
    shuffle=True)
model.save("model.h5")
# model = load_model("model.h5")

# グラフの表示
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()