import tensorflow as tf

def build(optimizer_config=None):
    summary_vars = []
    learning_rate = _create_learning_rate()
    summary_vars.append(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    return optimizer, summary_vars

def _create_learning_rate(learning_rate_config=None):
    learning_rate = tf.constant(0.001, dtype=tf.float32, name='learning_rate')
    return learning_rate
