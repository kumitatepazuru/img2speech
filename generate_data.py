import glob
import os
import shutil

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def make_image(font_path, dst_path, text, rot, font_size=32, font_color="white"):
    font_data = ImageFont.truetype(font_path, font_size)
    # get fontsize

    tmp = Image.new('RGBA', (1, 1), (0, 0, 0, 0))  # dummy for get text_size

    tmp_d = ImageDraw.Draw(tmp)
    text_size = tmp_d.textsize(text, font_data)
    i = font_size
    while text_size[0] > font_size - 5 or text_size[1] > font_size - 5:
        i -= 1
        font_data = ImageFont.truetype(font_path, i)
        text_size = tmp_d.textsize(text, font_data)
    # draw text

    img = Image.new('RGBA', [font_size] * 2, (0, 0, 0, 0))  # background: transparent

    img_d = ImageDraw.Draw(img)
    img_d.text((0, 0), text, fill=font_color, font=font_data)
    img = img.rotate(rot)
    img.save(dst_path)


try:
    shutil.rmtree("./img")
except FileNotFoundError:
    pass
os.mkdir("./img")
for rot in range(0, 180, 5):
    os.mkdir("./img/" + str(rot))
    t = tqdm(range(33, 127))
    for i in t:
        t.set_description(str(rot))
        os.mkdir("./img/" + str(rot) + "/" + str(i - 33))
        for font in glob.glob("./font/*"):
            try:
                make_image(font,
                           "./img/" + str(rot) + "/" + str(i - 33) + "/" + font.split("/")[-1] + "_" + str(
                               i - 33) + "_" + str(rot) + ".png", chr(i), rot)
            except OSError:
                pass
