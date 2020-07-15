# パッケージのインポート
import os

from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt
from tqdm import tqdm

import load_data

# data = np.load("train_datas.npz")
# train_images = data["train_images"]
# train_labels = data["train_labels"]
# test_images = data["test_images"]
# test_labels = data["test_labels"]

#
# # データセットの画像の確認
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(train_images[i], 'gray')
# plt.show()
#
# # データセットのラベルの確認
# print(train_labels[0:10])
#
train_paths = []
for root, dirs, files in tqdm(os.walk("./font")):
    train_paths += list(map(lambda n: root + "/" + n, files))

val_count = int(len(train_paths) * 0.01)

train_gen = load_data.Generator(
    train_paths[val_count:],
    batch_size=16384)
val_gen = load_data.Generator(
    train_paths[:val_count],
    batch_size=16384)

# モデルの作成
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(32**2,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(94, activation='softmax'))  # 出力層
# コンパイル
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['acc'])
# 学習
history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_gen.num_batches_per_epoch,
    validation_data=val_gen,
    validation_steps=val_gen.num_batches_per_epoch,
    epochs=100,
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

# # 評価
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))
#
# # 推論する画像の表示
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(test_images[i].reshape([int(math.sqrt(test_images.shape[1]))] * 2), 'gray')
# plt.show()
#
# # 推論したラベルの表示
# test_predictions = model.predict(test_images[0:1000])
# test_predictions = np.argmax(test_predictions, axis=1)
# print("".join(list(map(lambda n: chr(n+33), test_predictions.tolist()))))
