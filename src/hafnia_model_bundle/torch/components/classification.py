"""Classification components in torch: softmax and the top-1 'Classification' primitive."""

from typing import Any, Dict, List

import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification

from hafnia_model_bundle.core.definitions.classification import ClassificationFromLogits
from hafnia_model_bundle.torch.component import Component, register_component


def top1_classification(
    probabilities: torch.Tensor, task: TaskInfo, score_threshold: float = 0.0
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
    class_idx = int(probabilities.argmax())
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


@register_component
class TorchClassificationFromLogits(Component[ClassificationFromLogits]):
    def setup(self) -> None:
        self.task = self.context.bundle.get_task_by_name(self.params.task_name)

    def forward(self, logits: torch.Tensor) -> Dict[str, Any]:
        probabilities = logits.to(torch.float32).softmax(dim=-1)
        predictions = top1_classification(probabilities, task=self.task, score_threshold=self.params.score_threshold)
        return {"predictions": predictions}
