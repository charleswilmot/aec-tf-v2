import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import activations


class CriticPerScaleConv2D(keras.layers.Layer):
    def __init__(self, n_scales, filters, kernel_size, strides, padding='valid', activation=None,
            pool_size=(2, 2), pool_strides=(1, 1), pool_padding='valid'):
        super().__init__()
        # store parameters: #
        self._n_scales = n_scales
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._activation = activations.get(activation)
        self._pool_size = pool_size
        self._pool_strides = pool_strides
        self._pool_padding = pool_padding
        #
        self.flattens = [keras.layers.Flatten() for i in range(n_scales)]
        self.convs = [keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        ) for i in range(n_scales)]
        self.maxpools = [keras.layers.MaxPool2D(
            pool_size,
            strides=pool_strides,
            padding=pool_padding,
        ) for i in range(n_scales)]
        self.concat = keras.layers.Concatenate()

    def call(self, args):
        encodings_by_scale, actions = args
        visions = [
            conv(encodings_by_scale[scale_name])
            for conv, scale_name in zip(self.convs, sorted(encodings_by_scale))
        ]
        visions = [
            maxpool(vision)
            for maxpool, vision in zip(self.maxpools, visions)
        ]
        flat_visions = [
            flatten(vision)
            for flatten, vision in zip(self.flattens, visions)
        ]
        return self.concat(flat_visions + [actions])

    def get_config(self):
        return {
            "n_scales": self._n_scales,
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "padding": self._padding,
            "activation": activations.serialize(self._activation),
            "pool_size": self._pool_size,
            "pool_strides": self._pool_strides,
            "pool_padding": self._pool_padding,
        }


class PolicyPerScaleConv2D(keras.layers.Layer):
    def __init__(self, n_scales, filters, kernel_size, strides, padding='valid', activation=None,
            pool_size=(2, 2), pool_strides=(1, 1), pool_padding='valid'):
        super().__init__()
        # store parameters: #
        self._n_scales = n_scales
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._activation = activations.get(activation)
        self._pool_size = pool_size
        self._pool_strides = pool_strides
        self._pool_padding = pool_padding
        #
        self.flattens = [keras.layers.Flatten() for i in range(n_scales)]
        self.convs = [keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        ) for i in range(n_scales)]
        self.maxpools = [keras.layers.MaxPool2D(
            pool_size,
            strides=pool_strides,
            padding=pool_padding,
        ) for i in range(n_scales)]
        self.concat = keras.layers.Concatenate()

    def call(self, encodings_by_scale):
        visions = [
            conv(encodings_by_scale[scale_name])
            for conv, scale_name in zip(self.convs, sorted(encodings_by_scale))
        ]
        visions = [
            maxpool(vision)
            for maxpool, vision in zip(self.maxpools, visions)
        ]
        flat_visions = [
            flatten(vision)
            for flatten, vision in zip(self.flattens, visions)
        ]
        return self.concat(flat_visions)

    def get_config(self):
        return {
            "n_scales": self._n_scales,
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "padding": self._padding,
            "activation": activations.serialize(self._activation),
            "pool_size": self._pool_size,
            "pool_strides": self._pool_strides,
            "pool_padding": self._pool_padding,
        }


def downscale_10_tanh(x):
    return tf.tanh(x / 10)


def downscale_100_tanh(x):
    return tf.tanh(x / 100)


def downscale_500_tanh(x):
    return tf.tanh(x / 500)



custom_objects = {
    "CriticPerScaleConv2D": CriticPerScaleConv2D,
    "PolicyPerScaleConv2D": PolicyPerScaleConv2D,
    "downscale_10_tanh": downscale_10_tanh,
    "downscale_100_tanh": downscale_100_tanh,
    "downscale_500_tanh": downscale_500_tanh,
}


if __name__ == '__main__':
    import numpy as np

    a = keras.models.Sequential([CriticPerScaleConv2D(
        n_scales=2,
        filters=16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding='valid',
        activation=None,
        pool_size=(2, 2),
        pool_strides=(2, 2),
        pool_padding='valid',
    )])
    yaml = a.to_yaml()
    print(yaml)
    b = keras.models.model_from_yaml(yaml, custom_objects=custom_objects)

    inp = {
        "fine": np.zeros(shape=(32, 7, 7, 6)),
        "coarse": np.zeros(shape=(32, 7, 7, 6)),
    }
    actions = np.zeros(shape=(32, 4))

    out = a((inp, actions))
    out2 = b((inp, actions))



    encoder = keras.models.Sequential([
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding='valid',
            activation='relu',
        ),
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            activation='relu',
        ),
    ])
    decoder = keras.models.Sequential([
        keras.layers.Conv2D(
            filters=8 * 8 * 6,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            activation='relu',
        ),
    ])

    critic = keras.models.Sequential([
        CriticPerScaleConv2D(
            n_scales=2,
            filters=16,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            pool_size=(2, 2),
            pool_strides=(2, 2),
            pool_padding='valid',
        ),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(1),
    ])

    policy = keras.models.Sequential([
        PolicyPerScaleConv2D(
            n_scales=2,
            filters=16,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding='valid',
            activation='relu',
            pool_size=(2, 2),
            pool_strides=(2, 2),
            pool_padding='valid',
        ),
        keras.layers.Dense(200, activation=downscale_10_tanh),
        keras.layers.Dense(2, activation='tanh'),
    ])


    print("ENCODER")
    print(encoder.to_yaml())
    print("DECODER")
    print(decoder.to_yaml())
    print("CRITIC")
    print(critic.to_yaml())
    print("POLICY")
    print(policy.to_yaml())
