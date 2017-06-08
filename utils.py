import math
import tensorflow as tf
import numpy as np


def weight(name, shape, init='xavier', uniform_range=None):
    """ Initializes or reuses weight.
    Args:
        name : Variable name
        shape : Tensor shape
        init : Init mode. xavier / normal / uniform / he (default is 'he')
        uniform_range : Range of an uniform distribution if 'uniform' is chosen.
    Returns:
        tf.Variable
    """
    initializer = tf.constant_initializer()
    if init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        uniform_range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-uniform_range, uniform_range)

    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif init == 'uniform':
        if uniform_range is None:
            raise ValueError("range must not be None if uniform init is used.")
        initializer = tf.random_uniform_initializer(-uniform_range, uniform_range)

    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _get_dims(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def bias(name, dim, initial_value=0.0):
    """ Initializes bias parameter.
    :param name: Variable name
    :param dim: Tensor size (list or int)
    :param initial_value: Initial bias term
    :return: Variable
    """
    dims = dim if isinstance(dim, list) else [dim]
    return tf.get_variable(name, dims, initializer=tf.constant_initializer(initial_value))
