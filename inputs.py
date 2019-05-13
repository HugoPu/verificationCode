import functools
import tensorflow as tf

from protos import train_pb2, eval_pb2
from builders import dataset_builder, model_builder, image_resizer_builder
from utils import config_utils
from core import standard_fields as fields
from data_decoders import tf_example_decoder

SERVING_FED_EXAMPLE_KEY = 'serialized_example'

INPUT_BUILDER_UTIL_MAP = {
    'dataset_build': dataset_builder.build,
}

def _get_features_dict(input_dict):
    features = {
        fields.InputDataFields.image:
            input_dict[fields.InputDataFields.image],
        fields.InputDataFields.true_image_shape:
            input_dict[fields.InputDataFields.true_image_shape],
    }

    return features

def _get_label_dict(input_dict):
    features = {
        fields.InputDataFields.label:
            input_dict[fields.InputDataFields.label]
    }

    return features

def transform_input_data(tensor_dict,
                         model_preprocess_fn,
                         image_resizer_fn,):

    image = tensor_dict[fields.InputDataFields.image]
    preprocessed_resized_image, true_image_shape = model_preprocess_fn(
        tf.expand_dims(tf.to_float(image), axis=0))
    tensor_dict[fields.InputDataFields.image] = tf.squeeze(
        preprocessed_resized_image, axis=0)
    tensor_dict[fields.InputDataFields.true_image_shape] = tf.squeeze(
        true_image_shape, axis=0)

    return tensor_dict

def create_train_input_fn(train_config, train_input_config, model_config):
    def _train_input_fn(params=None):
        if not isinstance(train_config, train_pb2.TrainConfig):
            raise ValueError()
        def transform_and_pad_input_data_fn(tensor_dict):
            model = model_builder.build(model_config, is_training=True)
            image_resizer_config = config_utils.get_image_resizer_config(model_config)
            image_resizer_fn = image_resizer_builder.build(image_resizer_config)
            transform_data_fn = functools.partial(
                transform_input_data,
                model_preprocess_fn=model.preprocess,
                image_resizer_fn=image_resizer_fn)

            tensor_dict = transform_data_fn(tensor_dict)

            return (_get_features_dict(tensor_dict), _get_label_dict(tensor_dict))

        dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
            train_input_config,
            transform_input_data_fn=transform_and_pad_input_data_fn,
            batch_size=params['batch_size'] if params else train_config.batch_size
        )

        return dataset
    return _train_input_fn

def create_eval_input_fn(eval_config, eval_input_config, model_config):
    def _eval_input_fn(params=None):
        params = params or {}
        if not isinstance(eval_config, eval_pb2.EvalConfig):
            raise ValueError()
        def transform_and_pad_input_data_fn(tensor_dict):
            model = model_builder.build(model_config, is_training=False)
            image_resizer_config = config_utils.get_image_resizer_config(model_config)
            image_resizer_fn = image_resizer_builder.build(image_resizer_config)

            transform_data_fn = functools.partial(
                transform_input_data,
                model_preprocess_fn=model.preprocess,
                image_resizer_fn=image_resizer_fn)

            tensor_dict = transform_data_fn(tensor_dict)
            return (_get_features_dict(tensor_dict), _get_label_dict(tensor_dict))

        dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
            eval_input_config,
            transform_input_data_fn=transform_and_pad_input_data_fn,
            batch_size=params['batch_size'] if params else eval_config.batch_size
        )

        return dataset
    return _eval_input_fn

def create_predict_input_fn(model_config, predict_input_config):
    def _predict_input_fn(params=None):
        del params
        example = tf.placeholder(dtype=tf.string, shape=[], name='tf_example')
        model = model_builder.build(model_config, is_training=False)
        image_resizer_config = config_utils.get_image_resizer_config(model_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)

        transform_fn = functools.partial(
            transform_input_data,
            model_preprocess_fn=model.preprocess,
            image_resizer_fn=image_resizer_fn)

        decoder = tf_example_decoder.TfExampleDecoder()
        input_dict = transform_fn(decoder.decode(example))
        images = tf.to_float(input_dict[fields.InputDataFields.image])
        images = tf.expand_dims(images, axis=0)
        true_image_shape = tf.expand_dims(
            input_dict[fields.InputDataFields.true_image_shape], axis=0)

        return tf.estimator.export.ServingInputReceiver(features={
            fields.InputDataFields.image:images,
            fields.InputDataFields.true_image_shape:true_image_shape},
            receiver_tensors={SERVING_FED_EXAMPLE_KEY: example})
    return _predict_input_fn