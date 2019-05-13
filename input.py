from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from skimage.filters import threshold_local

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(max_captcha, min_captcha, char_set=number):
    captcha_text = []
    captcha_size = random.randint(min_captcha, max_captcha)
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长N个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 10
            return k
        k = ord(c) - 48
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector

# 生成一个训练batch
def get_next_batch(batch_size, IMAGE_HEIGHT, ):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(MAX_CAPTCHA, MIN_CAPTCHA)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

# 生成字符对应的验证码
def  gen_captcha_text_and_image(max_captcha, min_captcha):
    image = ImageCaptcha()

    captcha_text = random_captcha_text(max_captcha, min_captcha)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image