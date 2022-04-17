import tensorflow as tf


class SmoothL1Loss(tf.losses.Loss):
    """
        Implements Smooth L1 loss.
        Reference: https://arxiv.org/abs/1504.08083
    """
    def __init__(self, delta: float = 1.0, reduction='auto'):
        super().__init__(reduction=reduction, name="SmoothL1Loss")
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2

        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5
        )
        return tf.reduce_sum(loss, axis=-1)
