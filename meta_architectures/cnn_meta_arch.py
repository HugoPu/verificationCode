import tensorflow as tf
from utils import shape_utils

import logging

logger = logging.Logger(name='gen verification')

from core import standard_fields as fields

class CnnMetaArch():
    def __init__(self,
                 is_training,
                 image_resizer_fn,
                 config):
        self._is_trainging = is_training
        self._image_resizer_fn = image_resizer_fn
        self._config = config

    def preprocess(self, inputs):
        with tf.name_scope('Preprocessor'):
            outputs = shape_utils.static_or_dynamic_map_fn(
                self._image_resizer_fn,
                elems=inputs,
                dtype=[tf.float32, tf.int32])
            resized_inputs = outputs[0]
            true_image_shapes = outputs[1]

            return (resized_inputs, true_image_shapes)

    def predict(self, preprocessed_inputs):
        config = self._config
        w_alpha = config.w_alpha
        b_alpha = config.b_alpha
        dropout = 1 - config.keep_prob
        num_chars = config.num_chars
        char_length = len(config.chars)

        # 3 conv layer
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(
                preprocessed_inputs, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, dropout)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, dropout)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, dropout)

        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, dropout)

        with tf.name_scope('w_out'):
            w_out = tf.Variable(w_alpha * tf.random_normal([1024, num_chars * char_length]))

        with tf.name_scope('b_out'):
            b_out = tf.Variable(b_alpha * tf.random_normal([num_chars * char_length]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        # out = tf.nn.softmax(out)
        return out

    def loss(self, predictions, labels, scope=None):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=labels))
            tf.summary.scalar('loss', loss)
            self._loss = loss
        return loss

    # def updates(self):
    #     with tf.name_scope('optimizer'):
    #         optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self._loss)
    #     return optimizer

    def format_label(self, label):
        num_chars = self._config.num_chars
        char_length = len(self._config.chars)
        formated_label = tf.reshape(label, [-1, num_chars, char_length])

        return formated_label
