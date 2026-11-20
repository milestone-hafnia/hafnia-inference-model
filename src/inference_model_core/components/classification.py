"""Classification components."""

from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from inference_model_core.classification import softmax, top1_classification
from inference_model_core.component import NumpyComponent, register_component
from inference_model_core.model_info import ComponentInfo, ModelBundleInfo


class ClassificationFromLogitsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_name: str = Field(description="Task of the model info the logits belong to")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="No prediction below this probability")


@register_component
class ClassificationFromLogits(NumpyComponent):
    """'logits' (C,), one per class of the task -> the most likely class as a 'Classification' prediction."""

    type = "classification_from_logits"
    params_model = ClassificationFromLogitsParams
    params: ClassificationFromLogitsParams
    input_ports = ("logits",)
    output_ports = ("predictions",)

    def __init__(self, info: ComponentInfo, model_info: ModelBundleInfo):
        super().__init__(info, model_info)
        self.task = model_info.get_task_by_name(self.params.task_name)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        probabilities = softmax(np.asarray(inputs["logits"]))
        predictions = top1_classification(probabilities, task=self.task, score_threshold=self.params.score_threshold)
        return {"predictions": predictions}
