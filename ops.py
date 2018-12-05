import tensorflow as tf
import numpy as np


def spatial_shape(inputs, channels_first):

    inputs_shape = inputs.get_shape().as_list()

    return inputs_shape[2:] if channels_first else inputs_shape[1:-1]


def spatial_flatten(inputs, channels_first):

    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = ([-1, inputs_shape[1], np.prod(inputs_shape[2:])] if channels_first else
                     [-1, np.prod(inputs_shape[1:-1]), inputs_shape[-1]])

    return tf.reshape(inputs, outputs_shape)


def spatial_unflatten(inputs, spatial_shape, channels_first):

    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = ([-1, inputs_shape[1], spatial_shape[0], spatial_shape[1]] if channels_first else
                     [-1, spatial_shape[0], spatial_shape[1], inputs_shape[-1]])

    return tf.reshape(inputs, outputs_shape)
