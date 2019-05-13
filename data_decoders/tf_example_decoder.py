import tensorflow as tf

from core import standard_fields as fields

slim_example_decoder = tf.contrib.slim.tfexample_decoder

class TfExampleDecoder():
    def __init__(self):
        self.keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'label':
                tf.FixedLenFeature((66,), tf.float32),
        }
        self.items_to_handles = {
            fields.InputDataFields.image:slim_example_decoder.Image(),
            fields.InputDataFields.label:slim_example_decoder.Tensor('label')
        }

    def decode(self, tf_example_string_tensor):
        serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
        decoder = slim_example_decoder.TFExampleDecoder(
            self.keys_to_features, self.items_to_handles)
        keys = decoder.list_items()
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))

        return tensor_dict