"""Definitions of detection components: decoders that select detections, and conversion to 'Bbox' primitives."""

from typing import Optional

from pydantic import Field

from inference_model_core.definition import ComponentDefinition, Port
from inference_model_core.model_bundle import ComponentInfo

_DETECTION_OUTPUTS = (
    Port("boxes", "(K, 4) xyxy boxes in pixels, by decreasing score"),
    Port("scores", "(K,) scores"),
    Port("labels", "(K,) model labels"),
)


class AnchorNMS(ComponentDefinition):
    """Dense per-anchor predictions -> detections: score threshold, top-k and NMS per class.

    The output 'labels' are score columns.
    """

    type = "anchor_nms"
    input_ports = (
        Port("boxes", "(A, 4) decoded xyxy boxes per anchor"),
        Port("scores", "(A, C) class scores per anchor"),
    )
    output_ports = _DETECTION_OUTPUTS

    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    topk_candidates: int = Field(gt=0, description="Highest scoring candidates kept per class before NMS")
    max_detections: int = Field(gt=0)
    background_index: Optional[int] = Field(description="Score column of the background class, if any")


class QueryTopK(ComponentDefinition):
    """Detections of a DETR-style model: sigmoid class scores and top-k over (query, class). No NMS.

    The output 'boxes' are xyxy in network input pixels, 'labels' are logit columns.
    """

    type = "query_topk"
    input_ports = (
        Port("boxes", "(Q, 4) boxes as normalized (center x, center y, width, height)"),
        Port("logits", "(Q, C) class logits"),
        Port("transform", "'ImageTransform' of the resize, for the network input size"),
    )
    output_ports = _DETECTION_OUTPUTS

    num_select: int = Field(gt=0, description="Highest scoring (query, class) pairs kept")


class DetectionsToBboxes(ComponentDefinition):
    """Detections -> 'Bbox' predictions on the original image.

    Label i is class i of the task. Labels beyond the task's classes are dropped (e.g. RF-DETR's "no object"
    column). The boxes are in pixels of the network input when 'transform' is connected, or of the original
    'image' when that is connected instead.
    """

    type = "detections_to_bboxes"
    input_ports = (
        Port("boxes", "(N, 4) xyxy boxes in pixels"),
        Port("scores", "(N,) scores"),
        Port("labels", "(N,) model labels"),
        Port("transform", "'ImageTransform' from network input pixels to the original image", optional=True),
        Port("image", "The original image, for boxes that already are in its pixels", optional=True),
    )
    output_ports = (Port("predictions", "List of 'Bbox' predictions with normalized coordinates"),)

    task_name: str = Field(description="Task of the model info the detections belong to")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Detections below it are dropped")

    def check_connections(self, info: ComponentInfo) -> None:
        super().check_connections(info)
        if ("transform" in info.inputs) == ("image" in info.inputs):
            raise ValueError(f"Component '{info.name}' needs either the 'transform' or the 'image' input.")
