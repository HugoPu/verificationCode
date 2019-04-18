import tensorflow as tf

class Model(object):
    def __init__(self, hparams):
        self.hparams = hparams

        self.logits = self._build_graph()

    def _build_graph(self):
        hparams = self.hparams
        image_height = hparams.image_height
        image_width = hparams.image_width
        w_alpha = hparams.w_alpha
        b_alpha = hparams.b_alpha
        keep_prob = hparams.keep_prob
        char_length = hparams.char_length
        num_char = hparams.num_char

        X = tf.placeholder(tf.float32, [None, image_height * image_width])
        x = tf.reshape(X, shape=[-1, image_height, image_width, 1])

        # 3 conv layer
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, keep_prob)

        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)

        with tf.name_scope('w_out'):
            w_out = tf.Variable(w_alpha * tf.random_normal([1024, num_char * char_length]))

        with tf.name_scope('b_out'):
            b_out = tf.Variable(b_alpha * tf.random_normal([num_char * char_length]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        # out = tf.nn.softmax(out)
        return out

