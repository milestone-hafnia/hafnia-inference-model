"""Detection components: decoders that select detections, and conversion to 'Bbox' primitives."""

from typing import Any, Dict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from inference_model_core.component import NumpyComponent, register_component
from inference_model_core.detection import select_anchor_detections, select_query_detections, task_label_mask
from inference_model_core.model_info import ComponentInfo, ModelBundleInfo
from inference_model_core.preprocessing import ImageTransform
from inference_model_core.primitives import bboxes_from_input_xyxy


class AnchorNMSParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    topk_candidates: int = Field(gt=0, description="Highest scoring candidates kept per class before NMS")
    max_detections: int = Field(gt=0)
    background_index: int | None = Field(description="Score column of the background class, if any")


@register_component
class AnchorNMS(NumpyComponent):
    """Dense per-anchor predictions -> detections: score threshold, top-k and NMS per class.

    Inputs 'boxes' (A, 4) decoded xyxy boxes and 'scores' (A, C) class scores. Output 'labels' are score columns.
    """

    type = "anchor_nms"
    params_model = AnchorNMSParams
    params: AnchorNMSParams
    input_ports = ("boxes", "scores")
    output_ports = ("boxes", "scores", "labels")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        params = self.params
        boxes, scores, labels = select_anchor_detections(
            boxes=np.asarray(inputs["boxes"]),
            scores=np.asarray(inputs["scores"]),
            score_threshold=params.score_threshold,
            nms_iou_threshold=params.nms_iou_threshold,
            topk_candidates=params.topk_candidates,
            max_detections=params.max_detections,
            background_index=params.background_index,
        )
        return {"boxes": boxes, "scores": scores, "labels": labels}


class QueryTopKParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_select: int = Field(gt=0, description="Highest scoring (query, class) pairs kept")


@register_component
class QueryTopK(NumpyComponent):
    """Detections of a DETR-style model: sigmoid class scores and top-k over (query, class). No NMS.

    Inputs 'boxes' (Q, 4) normalized (center x, center y, width, height), 'logits' (Q, C) and the 'transform' of
    the resize. Output 'boxes' are xyxy in network input pixels, 'labels' are logit columns.
    """

    type = "query_topk"
    params_model = QueryTopKParams
    params: QueryTopKParams
    input_ports = ("boxes", "logits", "transform")
    output_ports = ("boxes", "scores", "labels")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        transform: ImageTransform = inputs["transform"]
        boxes, scores, labels = select_query_detections(
            boxes_cxcywh=np.asarray(inputs["boxes"]),
            logits=np.asarray(inputs["logits"]),
            num_select=self.params.num_select,
            input_hw=(transform.input_height, transform.input_width),
        )
        return {"boxes": boxes, "scores": scores, "labels": labels}


class DetectionsToBboxesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_name: str = Field(description="Task of the model info the detections belong to")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Detections below it are dropped")


@register_component
class DetectionsToBboxes(NumpyComponent):
    """Detections -> 'Bbox' predictions on the original image.

    Inputs 'boxes' (N, 4) xyxy pixels, 'scores' (N,) and 'labels' (N,), where label i is class i of the task.
    Labels beyond the task's classes are dropped (e.g. RF-DETR's "no object" column). The boxes are in pixels of
    the network input when 'transform' is connected, or of the original 'image' when that is connected instead.
    """

    type = "detections_to_bboxes"
    params_model = DetectionsToBboxesParams
    params: DetectionsToBboxesParams
    input_ports = ("boxes", "scores", "labels")
    optional_input_ports = ("transform", "image")
    output_ports = ("predictions",)

    def __init__(self, info: ComponentInfo, model_info: ModelBundleInfo):
        super().__init__(info, model_info)
        if ("transform" in info.inputs) == ("image" in info.inputs):
            raise ValueError(f"Component '{info.name}' needs either the 'transform' or the 'image' input.")
        self.task = model_info.get_task_by_name(self.params.task_name)
        self.n_classes = len(self.task.get_class_names() or [])

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if "transform" in inputs:
            transform = inputs["transform"]
        else:
            height, width = inputs["image"].shape[:2]
            transform = ImageTransform.identity(height, width)
        boxes, scores, labels = (np.asarray(inputs[port]) for port in ("boxes", "scores", "labels"))
        keep = task_label_mask(labels, self.n_classes)
        predictions = bboxes_from_input_xyxy(
            boxes[keep],
            scores[keep],
            labels[keep],
            task=self.task,
            transform=transform,
            score_threshold=self.params.score_threshold,
        )
        return {"predictions": predictions}
