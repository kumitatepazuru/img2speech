import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.utils import Sequence
from tqdm import tqdm


class Generator(Sequence):
    """Custom generator"""

    def __init__(self, data_paths, batch_size=1, width=28, height=28, font_size=28, num_of_class=74, rot=10):
        """construction

        :param data_paths: List of image file
        :param batch_size: Batch size
        :param width: Image width
        :param height: Image height
        :param num_of_class: Num of classes
        """

        # (tmp, _), (_, _) = cifar10.load_data()
        # print(tmp.shape)
        self.rot = rot
        self.data_paths = data_paths
        self.length = len(data_paths) * 94 * int(180 / self.rot) * 3
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.font_size = font_size
        self.num_of_class = num_of_class
        self.data_pos = [0, 0, 0, 0]
        self.font_data = ImageFont.truetype(self.data_paths[self.data_pos[0]], self.font_size)
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1
        # self.back_imgs = []
        # for i in range(50000):
        #     self.back_imgs.append(Image.fromarray(tmp[i]).convert("L").resize((self.width, self.height)))
        # del tmp

    # @profile
    def _load_data(self):
        text = chr(self.data_pos[2] + 48)
        font_color = "white"
        rot = self.data_pos[1] * 5

        try:
            # img = self.back_imgs[random.randint(0, 49999)]
            img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0)).convert("L")
            img_d = ImageDraw.Draw(img)
            img_d.text(((self.data_pos[3] - 1) * int(self.font_size / 3), 0), text, fill=font_color,
                       font=self.font_data)
            img = img.rotate(rot)
            img = np.array(img)
        except OSError:
            self.data_pos[2] = 0
            self.data_pos[0] += 1
            self.font_data = ImageFont.truetype(self.data_paths[self.data_pos[0]], self.font_size)
            self.data_pos[1] = 0
            img, _ = self._load_data()
        # img = img + np.random.rand(self.font_size, self.font_size)
        return np.where(img > 1, 1, img), self.data_pos[2]

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

            self.data_pos[3] += 1
            if self.data_pos[3] > 2:
                self.data_pos[2] += 1
                self.data_pos[3] = 0
                if self.data_pos[2] > 74:
                    self.data_pos[1] += 1
                    self.data_pos[2] = 0
                    if self.data_pos[1] > 180 / self.rot:
                        self.data_pos[0] += 1
                        self.font_data = ImageFont.truetype(self.data_paths[self.data_pos[0]], self.font_size)
                        self.data_pos[1] = 0
        # データセットの画像の前処理
        imgs = imgs.reshape((imgs.shape[0], imgs.shape[1] ** 2))
        # imgs = imgs.reshape((-1, self.height, self.width, 1))

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
        batch_size=512)
