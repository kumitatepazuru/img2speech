# パッケージのインポート
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("model.h5")

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    test_predictions = model.predict(test_images[0:1000])
    test_predictions = np.argmax(test_predictions, axis=1)
    print("".join(list(map(lambda n: chr(n + 33), test_predictions.tolist()))))
