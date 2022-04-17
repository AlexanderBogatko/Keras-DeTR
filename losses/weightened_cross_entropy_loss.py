from typing import Union

import numpy as np
import tensorflow as tf


class WeightenedCategoricalCrossentropy(tf.keras.losses.Loss):
    """
        Calculate categorical cross-entropy and multiply by class weights.
        Args:
            class_weights: array with weights for each class, shape (num_classes,),
            reduction: type of reduction to apply to loss
    """
    def __init__(self, class_weights: Union[list, np.ndarray], reduction: str = 'auto'):
        super().__init__(reduction=reduction)
        self.class_weights = tf.keras.backend.variable(class_weights)

    def call(self, y_true, y_pred):
        """
            Args:
                y_true: ground-truth tensor with one-hot encoded vectors, shape (batch, num_classes),
                y_pred: predicted tensor, shape (batch, num_classes)
            Returns:
                loss value
        """
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.keras.backend.log(y_pred) * self.class_weights
        return -tf.keras.backend.sum(loss, -1)
