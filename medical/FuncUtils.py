import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


def softmax(a, mask):
    """
    :param a: B*ML1*ML2
    :param mask: B*ML1*ML2
    """
    return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

