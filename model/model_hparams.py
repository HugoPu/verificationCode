import tensorflow as tf

def create_hparams(hparams_overrides=None):
    hparams = tf.contrib.training.HParams(
        load_pretrained=True
    )
    if hparams_overrides:
        hparams = hparams.parse(hparams_overrides)

    return hparams