import tensorflow as tf, tensorflow_addons as tfa
from typing import Union
from scipy.optimize import linear_sum_assignment


def center_form_to_corner_form(boxes: tf.Tensor) -> tf.Tensor:
	"""
	Converts boxes from center form (x, y, width, height) to corner form (x_min, y_min, x_max, y_max)
	Args:
		boxes: tensor with boxes in center form, shape (batch, num_boxes, 4), 4 means (x, y, width, height)
	Returns:
		tensor with boxes in corner form, shape (batch, num_boxes, 4)
	"""
	centers = boxes[..., :2]
	sizes = boxes[..., 2:]

	top_left = centers - sizes / 2
	bottom_right = centers + sizes / 2
	return tf.concat([top_left, bottom_right], axis=-1)


class GIoUAdapter(tf.keras.losses.Loss):
	"""
	https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/losses/giou_loss.py#L75)
	the tfa GIoU feeds boxes in the corner form while detr boxes take the center form. this adapter aims to dock them
	"""
	def __init__(self):
		super().__init__(name='giou_adapter')
		self.loss = tfa.losses.GIoULoss()

	def call(self, y_true, y_pred):
		y_true = center_form_to_corner_form(boxes=y_true)
		y_pred = center_form_to_corner_form(boxes=y_pred)
		return self.loss.call(y_true=y_true, y_pred=y_pred)


class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
	"""
	Calculate categorical cross-entropy and multiply by class weights.
	Args:
		class_weights: array with weights for each class, shape (num_classes,),
		reduction: type of reduction to apply to loss
	"""
	def __init__(self, class_weights: Union[list, tf.Variable], reduction: str = 'auto'):
		super().__init__(reduction=reduction)
		self.class_weights = class_weights

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


class Loss(tf.keras.losses.Loss):
	"""
	Implements loss for Detection Transformer, reference: https://arxiv.org/pdf/2005.12872.pdf
	Args:
		num_classes: total number of classes for detection (include 'background' (NoObject) class),
		max_boxes: maximal number of boxes to detect,
		background_id: index of 'background' class,
		background_class_weight: weight for 'background' (NoObject) class,
		cost_class_weight: weight for classification loss,
		cost_box_weight: weight for regression loss,
		cost_giou_weight: weight for Generalized IoU loss,
		reduction: type of reduction to apply to loss (look at tf.keras.losses.Loss doc for details)
	"""
	def __init__(self, num_classes: int,
				batch_size: int,
				max_boxes: int,
				background_id: int = 0,
				background_class_weight: float = 0.1,
				cost_class_weight: float = 1.,
				cost_box_weight: float = 5.,
				cost_giou_weight: float = 2.,
				reduction: str = 'auto'):
		super().__init__(reduction=reduction, name='detr_loss')
		self.background_id = background_id
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.max_boxes = max_boxes

		# cost weights
		self.cost_class_weight = cost_class_weight
		self.cost_box_weight = cost_box_weight
		self.cost_giou_weight = cost_giou_weight

		# define necessary losses
		class_weights = tf.Variable(tf.ones(shape=(num_classes,), dtype=tf.float32))
		class_weights[background_id].assign(background_class_weight)
		self.regression_loss = tf.keras.losses.MeanAbsoluteError()
		self.generalized_iou_loss = GIoUAdapter()
		self.classification_loss = WeightedCategoricalCrossentropy(class_weights=class_weights)

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

		# perform matching algorithm for each sample in batch
		total_loss = tf.zeros(self.max_boxes, dtype=tf.float32)
		for batch_id in range(self.batch_size):
			tf.autograph.experimental.set_loop_options(shape_invariants=[(total_loss, tf.TensorShape([None]))])
			# construct cost matrix
			cost_class = -tf.gather(predicted_scores[batch_id], gt_labels[batch_id], axis=1)  # (N[predictions], N[target])
			gt_boxes_expanded = gt_boxes[batch_id][tf.newaxis, ...]  # (1, N, 4)
			predicted_boxes_expanded = predicted_boxes[batch_id][:, tf.newaxis, :]  # (N, 1, 4)

			# (N[predictions], N[target])
			cost_boxes = self.regression_loss(y_true=gt_boxes_expanded, y_pred=predicted_boxes_expanded)
			cost_generalized_iou = self.generalized_iou_loss.call(y_true=gt_boxes[batch_id], y_pred=predicted_boxes[batch_id])  # (N[preds], N[gt])
			cost_matrix = self.cost_box_weight * cost_boxes + self.cost_class_weight * cost_class + self.cost_giou_weight * cost_generalized_iou

			# hungarian matching
			cost_matrix = tf.transpose(cost_matrix)  # (N[target], N[predictions])
			_, pred_indices = tf.py_function(linear_sum_assignment, inp=[cost_matrix], Tout=[tf.int32, tf.int32])

			# gather by matched indexes
			sample_pred_scores = tf.gather(predicted_scores[batch_id], pred_indices)
			sample_pred_boxes = tf.gather(predicted_boxes[batch_id], pred_indices)

			sample_gt_scores = tf.one_hot(gt_labels[batch_id], self.num_classes, dtype=tf.float32)
			sample_gt_boxes = gt_boxes[batch_id]

			# filter no-objects
			positive_mask = gt_labels[batch_id] != self.background_id
			sample_gt_boxes = tf.boolean_mask(sample_gt_boxes, positive_mask)
			sample_pred_boxes = tf.boolean_mask(sample_pred_boxes, tf.tile(positive_mask, [self.max_boxes]))

			# calculate loss values
			class_loss = self.classification_loss.call(y_true=sample_gt_scores, y_pred=sample_pred_scores)
			box_l1_loss = tf.math.reduce_mean(self.regression_loss(y_true=sample_gt_boxes, y_pred=sample_pred_boxes))
			box_giou_loss = tf.math.reduce_mean(self.generalized_iou_loss.call(y_true=sample_gt_boxes, y_pred=sample_pred_boxes))

			total_loss += self.cost_class_weight * class_loss + self.cost_box_weight * box_l1_loss + self.cost_giou_weight * box_giou_loss

		return total_loss
