"""Classification components."""

import numpy as np
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from pydantic import Field, PrivateAttr

from inference_model_core.classification import softmax, top1_classification
from inference_model_core.component import NumpyComponent, register_component
from inference_model_core.components.outputs import Predictions


@register_component
class ClassificationFromLogits(NumpyComponent):
    """'logits' (C,), one per class of the task -> the most likely class as a 'Classification' prediction."""

    type = "classification_from_logits"

    task_name: str = Field(description="Task of the model info the logits belong to")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="No prediction below this probability")

    _task: TaskInfo = PrivateAttr()

    def setup(self) -> None:
        self._task = self.context.model_info.get_task_by_name(self.task_name)

    def forward(self, logits: np.ndarray) -> Predictions:
        probabilities = softmax(np.asarray(logits))
        predictions = top1_classification(probabilities, task=self._task, score_threshold=self.score_threshold)
        return Predictions(predictions)  # type: ignore[arg-type]
