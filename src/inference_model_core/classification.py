"""Optional NumPy helpers for classification decoders: softmax and the top-1 'Classification' primitive."""

from typing import List

import numpy as np
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits.astype(np.float32) - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def top1_classification(
    probabilities: np.ndarray, task: TaskInfo, score_threshold: float = 0.0
) -> List[Classification]:
    """The most likely class as a 'Classification' prediction, or nothing if its probability is too low.

    Args:
        probabilities: (C,) class probabilities, one per class of 'task'.
        task: The model task.
        score_threshold: No prediction when the top probability is lower.
    """
    class_names = task.get_class_names()
    if class_names is None or len(class_names) != len(probabilities):
        raise ValueError(
            f"Got {len(probabilities)} class probabilities for task '{task.name}' with "
            f"{0 if class_names is None else len(class_names)} classes."
        )
    class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[class_idx])
    if confidence < score_threshold:
        return []
    return [
        Classification(
            class_idx=class_idx,
            class_name=class_names[class_idx],
            confidence=confidence,
            ground_truth=False,
            task_name=task.name,
        )
    ]
