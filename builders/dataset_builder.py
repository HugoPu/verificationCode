import os
import numpy as np
import tensorflow as tf
import glob
import random

from PIL import Image
from skimage.filters import threshold_local
from captcha.image import ImageCaptcha

from protos import input_reader_pb2
from core import standard_fields as fields


def generate_captcha_image(max_captcha,
                           min_captcha,
                           char_set):
    image = ImageCaptcha()

    captcha_text = []
    captcha_size = random.randint(min_captcha, max_captcha)
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)

    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def read_dataset(input_reader_config):
    config = input_reader_config.image_input_reader
    if not config.input_path:
        raise ValueError()
    image_paths = tf.gfile.Glob(config.input_path + '/*.jpg') + \
                  tf.gfile.Glob(config.input_path + '/*.JPG')

    features = []

    for path in image_paths:
        features.append({
            fields.InputDataFields.image: Image.open(path),
            fields.InputDataFields.label: os.path.basename(path).spilt('.')[0]
        })

    len_real = len(image_paths)
    len_captcha = int(len_real / input_reader_config.real_data_percent * (1 - len_real))
    for i in range(len_captcha):
        captcha_text, captcha_image = generate_captcha_image(
            input_reader_config.max_num_chars,
            input_reader_config.min_num_chars,
            input_reader_config.chars)
        features.append({
            fields.InputDataFields.image: captcha_image,
            fields.InputDataFields.label: captcha_text,
        })

    tensor_dict = tf.data.Dataset.from_tensor_slices(features)

    return tensor_dict

def build(input_reader_config, batch_size=None, transform_input_data=None):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError()

    if input_reader_config.WhichOneof('input_reader') == 'image_input_reader':
        def process_fn(value):
            if transform_input_data is not None:
                processed_tensors = transform_input_data(value)
            return processed_tensors

        num_parallel_map_calls = batch_size * input_reader_config.num_parallel_batches

        dataset = read_dataset(input_reader_config)

        dataset = dataset.map(process_fn, num_parallel_map_calls=num_parallel_map_calls)

        if batch_size:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)
        return dataset



    raise ValueError()

def image_preprocess(image, height, width):
    image = image.resize((width, height))
    image_np = load_limage_into_numpy_array(image)
    thresh = threshold_local(image_np, block_size=35)
    binary = image_np > thresh

    return binary

def load_limage_into_numpy_array(image):
  limage = image.convert('L')
  (im_width, im_height) = limage.size
  return np.array(limage.getdata()).reshape(
      (im_height, im_width)).astype(np.uint8)