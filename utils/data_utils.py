import os
import random
import shutil
import numpy as np
import tensorflow as tf

from PIL import Image
from captcha.image import ImageCaptcha

from core import standard_fields as fields

def text2vec(text, max_num_chars, char_set_len):
    text_len = len(text)
    if text_len > max_num_chars:
        raise ValueError('验证码最长N个字符')

    vector = np.zeros(max_num_chars * char_set_len)

    def char2pos(c):
        if c == '_':
            k = char_set_len - 1
            return k
        k = ord(c) - 48
        return k

    for i, c in enumerate(text):
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    return vector

def _generate_captcha_images(config, num_real_images):

    if os.path.exists(config.src_captcha_path):
        shutil.rmtree(config.src_captcha_path)

    os.mkdir(config.src_captcha_path)

    len_real = num_real_images
    real_data_percent = config.real_data_percent
    len_captcha = int(len_real / real_data_percent * (1 - real_data_percent))

    for i in range(len_captcha):
        # generate captcha image
        captcha_text = []
        captcha_size = random.randint(
            config.min_num_chars,
            config.max_num_chars)
        for i in range(captcha_size):
            char = random.choice(config.chars)
            captcha_text.append(char)
        captcha_text = ''.join(captcha_text)

        image = ImageCaptcha()
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)

        captcha_image.save(os.path.join(config.src_captcha_path, captcha_text + '.jpg'), 'jpeg')


def generate_tfrecords(config):
    real_image_paths = tf.gfile.Glob(config.src_real_path + '/*.jpg') + \
                  tf.gfile.Glob(config.src_real_path + '/*.JPG')

    captcha_image_paths = []
    if config.is_generate_captcha:
        _generate_captcha_images(config, len(real_image_paths))
        captcha_image_paths = tf.gfile.Glob(config.src_captcha_path + '/*.jpg')

    # image_paths = real_image_paths + captcha_image_paths
    image_paths = captcha_image_paths

    writer = tf.python_io.TFRecordWriter(config.output_path)

    for path in image_paths:
        with tf.gfile.GFile(path, 'rb') as fid:
            encoded_jpg = fid.read()
        label_str = os.path.basename(path).split('.')[0]
        label = text2vec(label_str, config.max_num_chars, len(config.chars) + 1)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
            'image/format':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpg'.encode('utf8')])),
            'label':
                tf.train.Feature(float_list=tf.train.FloatList(value=label))
        }))

        writer.write(tf_example.SerializeToString())
    writer.close()


