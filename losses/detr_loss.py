import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from .generalized_iou_loss import GeneralizedIoU
from .weightened_cross_entropy_loss import WeightenedCategoricalCrossentropy


class DeTRLoss(tf.keras.losses.Loss):
    """
        Implements loss for Detection Transformer, reference: https://arxiv.org/pdf/2005.12872.pdf
        Args:
            num_classes: total number of classes for detection (include 'background' (NoObject) class),
            background_id: index of 'background' class,
            background_class_weight: weight for 'background' (NoObject) class,
            cost_class_weight: weight for classification loss,
            cost_box_weight: weight for regression loss,
            cost_giou_weight: weight for Generalized IoU loss,
            reduction: type of reduction to apply to loss (look at tf.keras.losses.Loss doc for details)
    """

    def __init__(self, num_classes: int,
                 background_id: int,
                 background_class_weight: float = 0.1,
                 cost_class_weight: float = 1.,
                 cost_box_weight: float = 5.,
                 cost_giou_weight: float = 2.,
                 reduction: str = 'auto'):
        super().__init__(reduction=reduction, name='DeTRLoss')
        self.background_id = background_id
        self.num_classes = num_classes

        # Cost weights
        self.cost_class_weight = cost_class_weight
        self.cost_box_weight = cost_box_weight
        self.cost_giou_weight = cost_giou_weight

        # Define necessary losses
        class_weights = np.ones(shape=(num_classes,), dtype=np.float32)
        class_weights[background_id] = background_class_weight
        self.regression_loss = tf.keras.losses.MeanAbsoluteError(reduction='none')
        self.generalized_iou_loss = GeneralizedIoU(reduction='none')
        self.classification_loss = WeightenedCategoricalCrossentropy(class_weights=class_weights)

    def call(self, y_true, y_pred):
        """
            Args:
                y_true: ground-truth tensor, shape (batch, num_boxes, 5), 5 means (label_id, x, y, width, height)
                y_pred: predicted tensor, shape (batch, num_boxes, num_classes + 4), 4 means (x, y, width, height)
        """
        gt_boxes = y_true[..., 1:]  # (batch, num_boxes, 4)
        gt_labels = tf.cast(y_true[..., 0], dtype=tf.int32)  # (batch, num_boxes)

        predicted_boxes = y_pred[..., self.num_classes:]  # (batch, num_boxes, 4)
        predicted_scores = y_pred[..., :self.num_classes]  # (batch, num_boxes, num_classes)

        # Perform matching algorithm for each sample in batch
        total_loss = 0
        for batch_id in range(gt_boxes.shape[0]):

            # Construct cost matrix
            cost_class = -tf.gather(predicted_scores[batch_id], gt_labels[batch_id], axis=1)  # (N[preds], N[gt])

            gt_boxes_expanded = gt_boxes[batch_id][tf.newaxis, ...]  # (1, N, 4)
            predicted_boxes_expanded = gt_boxes[batch_id][:, tf.newaxis, :]  # (N, 1, 4)

            # (N[preds], N[gt])
            cost_boxes = self.regression_loss(y_true=gt_boxes_expanded, y_pred=predicted_boxes_expanded)

            cost_generalized_iou = self.generalized_iou_loss(
                y_true=gt_boxes[batch_id][np.newaxis, ...],
                y_pred=predicted_boxes[batch_id][:, np.newaxis, :]
            )  # (N[preds], N[gt])

            cost_matrix = self.cost_box_weight * cost_boxes + \
                          self.cost_class_weight * cost_class + \
                          self.cost_giou_weight * cost_generalized_iou

            # Hungarian matching
            cost_matrix = np.transpose(cost_matrix.numpy(), axes=[1, 0])  # (N[gt], N[preds])
            _, pred_indices = linear_sum_assignment(cost_matrix)

            # Gather by matched indexes
            sample_pred_scores = tf.gather(predicted_scores[batch_id], pred_indices)
            sample_pred_boxes = tf.gather(predicted_boxes[batch_id], pred_indices)

            sample_gt_scores = tf.one_hot(gt_labels[batch_id], self.num_classes)
            sample_gt_boxes = gt_boxes[batch_id]

            # Filter no-objects
            positive_mask = gt_labels[batch_id] != self.background_id
            sample_gt_boxes = tf.boolean_mask(sample_gt_boxes, positive_mask)
            sample_pred_boxes = tf.boolean_mask(sample_pred_boxes, positive_mask)

            # Calculate loss values
            class_loss = self.classification_loss(y_true=sample_gt_scores, y_pred=sample_pred_scores)
            box_l1_loss = tf.reduce_mean(self.regression_l1_loss(y_true=sample_gt_boxes, y_pred=sample_pred_boxes))
            box_giou_loss = tf.reduce_mean(self.generalized_iou_loss(y_true=sample_gt_boxes, y_pred=sample_pred_boxes))

            total_loss += self.cost_class_weight * class_loss + \
                          self.cost_box_weight * box_l1_loss + \
                          self.cost_giou_weight * box_giou_loss

        return total_loss
