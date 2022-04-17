from typing import Dict

import tensorflow as tf

from smooth_l1_loss import SmoothL1Loss
from generalized_iou_loss import GeneralizedIoU
from weightened_cross_entropy_loss import WeightenedCategoricalCrossentropy
from detr_loss import DeTRLoss


def get_detr_losses() -> Dict[str, type(tf.keras.losses.Loss)]:
    return {'detr_loss': DeTRLoss}
