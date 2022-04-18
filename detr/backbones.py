from typing import Dict, Type

import tensorflow as tf


class Backbone:
    """
        Base class for backbones.
    """
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    @classmethod
    def get_name(cls):
        return cls.__name__

    def __call__(self) -> tf.keras.Model:
        raise NotImplementedError('Method must be implemented in child class!')


class ResNet50(Backbone):
    def __call__(self) -> tf.keras.Model:
        return tf.keras.applications.ResNet50(include_top=False, input_shape=self.input_shape)


class ResNet101(Backbone):
    def __call__(self) -> tf.keras.Model:
        return tf.keras.applications.ResNet101(include_top=False, input_shape=self.input_shape)


def get_available_backbones() -> Dict[str, Type[Backbone]]:
    return {backbone.get_name(): backbone for backbone in [ResNet50, ResNet101]}
