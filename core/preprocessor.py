import tensorflow as tf

from utils import shape_utils

def resize_image(image,
                 masks=None,
                 new_height=600,
                 new_width=1024,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):

    with tf.name_scope(
            'ResizeImage',
            values=[image, new_height, new_width, method, align_corners]):
        new_image = tf.image.resize_images(
            image,
            tf.stack([new_height, new_width]),
            method=method,
            align_corners=align_corners)
        image_shape = shape_utils.combined_static_and_dynamic_shape(image)
        result = [new_image]
        result.append(tf.stack([new_height, new_width, image_shape[2]]))

        return result

def rgb_to_gray(image):
    return _rgb_to_grayscale(image)

def _rgb_to_grayscale(images, name=None):
    with tf.name_scope(name, 'rgb_to_grayscale', [images]) as name:
        images = tf.convert_to_tensor(images, name='images')
        orig_dtype = images.dtype
        flt_image = tf.image.convert_image_dtype(images, tf.float32)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        rank_1 = tf.expand_dims(tf.rank(images) - 1, 0)
        gray_float = tf.reduce_sum(flt_image * rgb_weights, rank_1, keep_dims=True)
        gray_float.set_shape(images.get_shape()[:-1].concatenate([1]))
        return tf.image.convert_image_dtype(gray_float, orig_dtype, name=name)
