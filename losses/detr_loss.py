import tensorflow as tf, tensorflow_addons as tfa
from scipy.optimize import linear_sum_assignment


class Loss(tf.keras.losses.Loss):
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
				background_id: int = 0,
				cost_class_weight: float = 1.,
				cost_box_weight: float = 5.,
				cost_giou_weight: float = 2.,
				reduction: str = 'auto'):
		super().__init__(reduction=reduction, name='detr_loss')
		self.background_id = background_id
		self.num_classes = num_classes

		# cost weights
		self.cost_class_weight = cost_class_weight
		self.cost_box_weight = cost_box_weight
		self.cost_giou_weight = cost_giou_weight

		# define necessary losses
		self.regression_loss = tf.keras.losses.MeanAbsoluteError()
		self.generalized_iou_loss = tfa.losses.GIoULoss(mode='giou')
		self.classification_loss = tf.keras.losses.CategoricalCrossentropy()

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
		total_loss = tf.constant(0, dtype=tf.float32)
		for batch_id in range(tf.shape(gt_boxes)[0]):
			# construct cost matrix
			cost_class = -tf.gather(predicted_scores[batch_id], gt_labels[batch_id], axis=1)  # (N[predictions], N[target])
			# gt_boxes_expanded = gt_boxes[batch_id][tf.newaxis, ...]  # (1, N, 4)
			# predicted_boxes_expanded = gt_boxes[batch_id][:, tf.newaxis, :]  # (N, 1, 4)

			# (N[predictions], N[target])
			cost_boxes = self.regression_loss(y_true=gt_boxes[batch_id], y_pred=predicted_boxes[batch_id])
			cost_generalized_iou = self.generalized_iou_loss(y_true=gt_boxes[batch_id], y_pred=predicted_boxes[batch_id])  # (N[preds], N[gt])
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
			sample_pred_boxes = tf.boolean_mask(sample_pred_boxes, positive_mask)

			# calculate loss values
			class_loss = self.classification_loss(y_true=sample_gt_scores, y_pred=sample_pred_scores)
			box_l1_loss = tf.math.reduce_mean(self.regression_loss(y_true=sample_gt_boxes, y_pred=sample_pred_boxes))
			box_giou_loss = tf.math.reduce_mean(self.generalized_iou_loss(y_true=sample_gt_boxes, y_pred=sample_pred_boxes))

			total_loss += self.cost_class_weight * class_loss + self.cost_box_weight * box_l1_loss + self.cost_giou_weight * box_giou_loss

		return total_loss
