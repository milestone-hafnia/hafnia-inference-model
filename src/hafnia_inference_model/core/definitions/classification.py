"""Definitions of classification components."""

from pydantic import Field

from hafnia_inference_model.core.definition import ComponentDefinition, Port


class ClassificationFromLogits(ComponentDefinition):
    """Class logits -> the most likely class as a 'Classification' prediction."""

    type = "classification_from_logits"
    input_ports = (Port("logits", "(C,) logits, one per class of the task"),)
    output_ports = (Port("predictions", "List with the 'Classification' prediction, empty below the threshold"),)

    task_name: str = Field(description="Task of the model info the logits belong to")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="No prediction below this probability")
