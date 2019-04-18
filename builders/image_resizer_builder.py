import functools
import tensorflow as tf

from protos import image_resizer_pb2
from core import preprocessor

def _tf_resize_method(resize_method):
    dict_method = {
        image_resizer_pb2.BILINEAR:
            tf.image.ResizeMethod.BILINEAR,
        image_resizer_pb2.BICUBIC:
            tf.image.ResizeMethod.BICUBIC
    }
    if resize_method in dict_method:
        return dict_method[resize_method]
    else:
        raise ValueError('Unknown resize_method')

def build(image_resizer_config):
    if not isinstance(image_resizer_config, image_resizer_pb2.ImageResizer):
        raise ValueError()

    image_resizer_oneof = image_resizer_config.WhichOneof('image_resizer_oneof')
    if image_resizer_oneof == 'fixed_shape_resizer':
        fixed_shape_resizer_config = image_resizer_config.fixed_shape_resizer
        method = _tf_resize_method(fixed_shape_resizer_config.resize_method)
        image_resizer_fn = functools.partial(
            preprocessor.resize_image,
            new_height=fixed_shape_resizer_config.height,
            new_width=fixed_shape_resizer_config.width,
            method=method)
        if not fixed_shape_resizer_config.convert_to_grayscale:
            return image_resizer_fn
    else:
        raise ValueError()