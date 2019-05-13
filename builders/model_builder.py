from protos import model_pb2
from builders import image_resizer_builder
from meta_architectures import cnn_meta_arch

def build(model_config, is_training, add_summaries=True):
    if not isinstance(model_config, model_pb2.Model):
        raise ValueError()
    meta_architecture = model_config.WhichOneof('model')
    if meta_architecture == 'cnn':
        return _build_cnn_model(model_config.cnn, is_training, add_summaries)
    else:
        raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))

def _build_cnn_model(cnn_config, is_training, add_summaries):
    image_resizer_fn = image_resizer_builder.build(cnn_config.image_resizer)
    cnn_meta_arch_fn = cnn_meta_arch.CnnMetaArch
    return cnn_meta_arch_fn(
        is_training=is_training,
        image_resizer_fn=image_resizer_fn,
        config=cnn_config)