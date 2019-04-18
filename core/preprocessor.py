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
