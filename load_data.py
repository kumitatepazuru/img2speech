import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.utils import Sequence
from tqdm import tqdm


class Generator(Sequence):
    """Custom generator"""

    def __init__(self, data_paths, batch_size=1, width=32, height=32, font_size=32, num_of_class=94):
        """construction

        :param data_paths: List of image file
        :param batch_size: Batch size
        :param width: Image width
        :param height: Image height
        :param num_of_class: Num of classes
        """

        self.data_paths = data_paths
        self.length = len(data_paths) * 94 * int(180 / 5)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.font_size = font_size
        self.num_of_class = num_of_class
        self.data_pos = [0, 0, 0]
        self.font_data = ImageFont.truetype(self.data_paths[self.data_pos[0]], self.font_size - 5)
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1

    def _load_data(self):
        text = chr(self.data_pos[2] + 33)
        font_color = "white"
        rot = self.data_pos[1] * 5

        try:
            img = Image.new('RGBA', [self.font_size] * 2, (0, 0, 0, 0)).convert("L")  # background: transparent
            img_d = ImageDraw.Draw(img)
            img_d.text((0, 0), text, fill=font_color, font=self.font_data)
            img = img.rotate(rot)

            self.data_pos[2] += 1
            if self.data_pos[2] > 93:
                self.data_pos[1] += 1
                self.data_pos[2] = 0
                if self.data_pos[1] > 180 / 5:
                    self.data_pos[0] += 1
                    self.font_data = ImageFont.truetype(self.data_paths[self.data_pos[0]], self.font_size)
                    self.data_pos[1] = 0
            img = np.array(img)
        except OSError:
            self.data_pos[2] = 0
            self.data_pos[0] += 1
            self.font_data = ImageFont.truetype(self.data_paths[self.data_pos[0]], self.font_size)
            self.data_pos[1] = 0
            img, _ = self._load_data()
        return img + np.random.rand(self.font_size, self.font_size), self.data_pos[2]

    def __getitem__(self, idx) -> np.array:
        """Get batch data

        :param idx: Index of batch

        :return imgs: numpy array of images
        :return labels: numpy array of label
        """

        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        imgs = np.empty((end_pos - start_pos + 1, self.height, self.width), dtype=np.float32)
        labels = np.zeros((end_pos - start_pos + 1, self.num_of_class), dtype=np.int16)

        for i in range(end_pos - start_pos + 1):
            img, label = self._load_data()
            imgs[i, :] = img
            labels[i][label] = 1
        # データセットの画像の前処理
        imgs = imgs.reshape((imgs.shape[0], imgs.shape[1] ** 2))

        return imgs, labels

    def __len__(self):
        """Batch length"""

        return self.num_batches_per_epoch

    def on_epoch_end(self):
        """Task when end of epoch"""
        pass


if __name__ == '__main__':
    train_paths = []
    for root, dirs, files in tqdm(os.walk("./font")):
        train_paths += list(map(lambda n: root + "/" + n, files))
    tmp = Generator(
        train_paths,
        batch_size=4096)
    a = tmp[0]
