from typing import List, Union

import cv2
import numpy as np


def draw_boxes(
        image: np.ndarray,
        boxes: np.ndarray,
        labels: Union[np.ndarray, List[str]],
        box_color: tuple = (0, 255, 255),
        text_color: tuple = (0, 0, 0),
        text_font: int = cv2.FONT_HERSHEY_SIMPLEX,
        text_font_size: float = 0.6,
        text_thickness: int = 1
    ) -> np.ndarray:
    """
        Bounding boxes visualization
        Args:
            image: uint8 numpy array with shape (h, w, c)
            boxes: int32 array of corner-form boxes with shape (num_boxes, 4), 4 - x_min, y_min, x_max, y_max
            labels: array with label names, length must be equal num_boxes
            box_color: bounding boxes color
            text_color: color of text (black by default)
            text_font: font of text
            text_font_size: size of text font
            text_thickness: text thickness
        Returns:
            image with drawn boxes
    """

    assert len(boxes) == len(labels), '`boxes` and `labels` must have same length!'
    assert boxes.dtype == np.int32, '`boxes` must have int32 format!'

    result = image.copy()
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box

        # Draw bounding box
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color=box_color, thickness=1)

        # Get size of text
        (text_w, text_h), _ = cv2.getTextSize(label, text_font, text_font_size, text_thickness)

        # Draw filled box
        cv2.rectangle(result, (x_min, y_min - text_h), (x_min + text_w, y_min), color=box_color, thickness=-1)

        # Draw text
        cv2.putText(result, label, (x_min, y_min), text_font, text_font_size, text_color, text_thickness)

    return result
