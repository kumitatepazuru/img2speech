from model import larger_model
from emnist import extract_training_samples, extract_test_samples

(train_images, train_labels), (test_images, test_labels) = extract_training_samples("byclass"), \
                                                           extract_test_samples('byclass')

from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# データセットの画像の前処理
train_images = train_images.reshape((-1, train_images.shape[1], train_images.shape[2], 1))
test_images = test_images.reshape((-1, train_images.shape[1], train_images.shape[2], 1))

# データセットのラベルの前処理
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# モデルの作成
model = larger_model()

# 学習
history = model.fit(train_images, train_labels, batch_size=500,
                    epochs=5, validation_split=0.2)

# グラフの表示
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

# 評価
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))

# 推論する画像の表示
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(test_images[i].reshape((28, 28)), 'gray')
plt.show()

# 推論したラベルの表示
test_predictions = model.predict(test_images[0:10])
test_predictions = np.argmax(test_predictions, axis=1)
print(test_predictions)
