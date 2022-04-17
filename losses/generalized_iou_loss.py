import tensorflow as tf


class GeneralizedIoU(tf.keras.losses.Loss):
    """
        Implements Generalized IoU loss.
        Reference: https://giou.stanford.edu/
    """

    def _center_form_to_corner_form(self, boxes: tf.Tensor) -> tf.Tensor:
        """
            Converts boxes from center form (x, y, width, height) to corner form (x_min, y_min, x_max, y_max)
            Args:
                boxes: tensor with boxes in center form, shape (batch, num_boxes, 4), 4 means (x, y, width, height)
            Returns:
                tensor with boxes in corner form, shape (batch, num_boxes, 4)
        """
        centers = boxes[..., :2]
        sizes = boxes[..., 2:]

        top_left_coordinates = centers - sizes / 2
        bottom_right_coordinates = centers + sizes / 2

        return tf.concat([top_left_coordinates, bottom_right_coordinates], axis=-1)

    def _calculate_areas(self, boxes: tf.Tensor) -> tf.Tensor:
        """
            Args:
                boxes: tensor with boxes in corner form (normalized between 0 and 1), shape (batch, num_boxes, 4),
                        4 means (x_min, y_min, x_max, y_max)
            Returns:
                tensor with areas for each box, shape (batch, num_boxes)
        """
        # Calculate boxes width and height
        area_width_height = tf.clip_by_value(boxes[..., 2:] - boxes[..., :2], clip_value_min=0, clip_value_max=1)
        return area_width_height[..., 0] * area_width_height[..., 1]  # Calculate box area (width * height)

    def call(self, y_true, y_pred):
        """
            Calculate Generalized Iou loss.
            Args:
                y_true: (batch, num_boxes, 4) - ground-truth boxes in center form
                y_pred: (batch, num_boxes, 4) - predicted boxes in center form
            Returns:
                loss value
        """
        y_true = self._center_form_to_corner_form(boxes=y_true)
        y_pred = self._center_form_to_corner_form(boxes=y_pred)

        true_area = self._calculate_areas(boxes=y_true)
        pred_area = self._calculate_areas(boxes=y_pred)

        # Calculate intersections
        intersection_left_top = tf.maximum(y_true[..., :2], y_pred[..., :2])
        intersection_right_bottom = tf.minimum(y_true[..., 2:], y_pred[..., 2:])
        intersection_width_height = tf.clip_by_value(
            intersection_right_bottom - intersection_left_top,
            clip_value_min=0,
            clip_value_max=1
        )
        intersection = intersection_width_height[..., 0] * intersection_width_height[..., 1]

        # Calculate smallest enclosing area (look at reference for more details)
        smallest_enclosing_box_left_top = tf.minimum(y_true[..., :2], y_pred[..., :2])
        smallest_enclosing_box_right_bottom = tf.maximum(y_true[..., 2:], y_pred[..., 2:])
        smallest_enclosing_width_height = tf.clip_by_value(
            smallest_enclosing_box_right_bottom - smallest_enclosing_box_left_top,
            clip_value_min=0,
            clip_value_max=1
        )
        smallest_enclosing_area = smallest_enclosing_width_height[..., 0] * smallest_enclosing_width_height[..., 1]

        # Calculate loss value
        union = true_area + pred_area - intersection
        iou = intersection / (union + tf.keras.backend.epsilon())
        generalized_iou = iou - (smallest_enclosing_area - union) / (smallest_enclosing_area + tf.keras.backend.epsilon())

        return 1 - generalized_iou
