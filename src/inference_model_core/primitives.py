"""Helpers to convert decoded model outputs into hafnia primitives."""

from typing import List

import numpy as np
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox

from inference_model_core.preprocessing import ImageTransform


def bboxes_from_input_xyxy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    task: TaskInfo,
    transform: ImageTransform,
    score_threshold: float = 0.0,
) -> List[Bbox]:
    """Convert decoded detections to 'Bbox' predictions on the original image.

    Args:
        boxes_xyxy: (N, 4) boxes in pixel coordinates of the model input.
        scores: (N,) confidence scores.
        class_indices: (N,) indices into 'task.classes'.
        task: The model task the boxes belong to.
        transform: How the original image maps to the model input.
        score_threshold: Detections with a lower score are dropped.
    """
    class_names = task.get_class_names()
    if class_names is None:
        raise ValueError(f"Task '{task.name}' has no classes defined.")

    boxes = transform.input_xyxy_to_normalized(boxes_xyxy)
    scores = np.asarray(scores).reshape(-1)
    class_indices = np.asarray(class_indices).reshape(-1)
    if not (len(boxes) == len(scores) == len(class_indices)):
        raise ValueError(
            f"Got {len(boxes)} boxes, {len(scores)} scores and {len(class_indices)} class indices. Lengths must match."
        )

    bboxes = []
    for (x0, y0, x1, y1), score, class_idx in zip(boxes, scores, class_indices, strict=True):
        if score < score_threshold or x1 <= x0 or y1 <= y0:
            continue
        class_idx = int(class_idx)
        if not 0 <= class_idx < len(class_names):
            raise ValueError(f"Class index {class_idx} is out of range for task '{task.name}' ({len(class_names)}).")
        bboxes.append(
            Bbox(
                top_left_x=float(x0),
                top_left_y=float(y0),
                width=float(x1 - x0),
                height=float(y1 - y0),
                class_idx=class_idx,
                class_name=class_names[class_idx],
                confidence=float(score),
                ground_truth=False,
                task_name=task.name,
            )
        )
    return bboxes
