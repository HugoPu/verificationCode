import tensorflow as tf

from google.protobuf import text_format
from protos import train_pb2
from protos import pipeline_pb2

def get_image_resizer_config(model_config):
    meta_architecture = model_config.WhichOneof('model')
    if meta_architecture == 'cnn':
        return model_config.cnn.image_resizer
    else:
        raise ValueError()

def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    if config_override:
        text_format.Merge(config_override, pipeline_config)
    return create_configs_from_pipeline_proto(pipeline_config)

def create_configs_from_pipeline_proto(pipeline_config):
    configs = {}
    configs["model"] = pipeline_config.model
    configs["train_config"] = pipeline_config.train_config
    configs["train_input_config"] = pipeline_config.train_input_reader
    configs["eval_config"] = pipeline_config.eval_config
    configs["eval_input_configs"] = pipeline_config.eval_input_reader
    # if configs["eval_input_configs"]:
    #     configs["eval_input_configs"] = configs["eval_input_configs"][0]
    # if pipeline_config.HasField("graph_rewriter"):
    #     configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

    return configs

# def merge_external_params_with_configs(configs, hparams=None, kwargs_dict=None):
#     if kwargs_dict is None:
#         kwargs_dict = {}
#     if hparams:
#         kwargs_dict.update(hparams.values)
#     for key, value in kwargs_dict.items():
#         tf.logging.infor('Maybe overwriting %s:%s', key, value)
#         if value == "" or value is None:
#             continue
#         elif _maybe_update_config_with_key_value(configs, key, value):
#             continue
#         elif _is_generic(key):
#             _update_generic(configs, key, value)
#         else:
#             tf.logging.info("Ignoring config override key:%s", key)
#     return configs
