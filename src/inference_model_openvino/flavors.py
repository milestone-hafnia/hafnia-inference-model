"""OpenVINO flavors. Registered through the 'hafnia.inference_models' entry point (see pyproject.toml).

The postprocessing mirrors 'inference_model_onnxruntime.flavors' on purpose: engine packages share
nothing but the core module (docs/design.md, 3.4). Cross-engine parity tests keep the copies in line.
"""

from typing import List

from hafnia.dataset.primitives import Primitive
from pydantic import BaseModel, ConfigDict, Field

from inference_model_core import PreprocessMeta, RawTensors, register_flavor
from inference_model_core.detection import map_labels, select_anchor_detections
from inference_model_core.primitives import bboxes_from_input_xyxy
from inference_model_openvino.engine import OpenVINOModel


class AnchorBoxesNMSParams(BaseModel):
    """Parameters of the 'anchor_boxes_nms' decoder: per-anchor boxes and class scores, NMS outside the graph."""

    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    topk_candidates: int = Field(gt=0, description="Highest scoring candidates kept per class before NMS")
    max_detections: int = Field(gt=0)
    background_index: int | None = Field(description="Score column of the background class, if any")
    label_to_class_idx: List[int] = Field(description="Score column -> task class index, -1 = ignored")


class BoxesScoresLabelsParams(BaseModel):
    """Parameters of the 'boxes_scores_labels' decoder: final detections from the graph."""

    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(ge=0.0, le=1.0)
    label_to_class_idx: List[int] = Field(description="Model label -> task class index, -1 = ignored")


@register_flavor
class OpenVINOTorchvisionSSD(OpenVINOModel):
    """torchvision SSD models exported as network only ('model_exporters.torchvision_detection')."""

    framework = "torchvision"
    decoder = "anchor_boxes_nms"
    params_model = AnchorBoxesNMSParams
    params: AnchorBoxesNMSParams

    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        boxes, scores, labels = select_anchor_detections(
            boxes=outputs["boxes"][0],
            scores=outputs["scores"][0],
            score_threshold=self.params.score_threshold,
            nms_iou_threshold=self.params.nms_iou_threshold,
            topk_candidates=self.params.topk_candidates,
            max_detections=self.params.max_detections,
            background_index=self.params.background_index,
        )
        class_indices, keep = map_labels(labels, self.params.label_to_class_idx)
        return bboxes_from_input_xyxy(  # type: ignore[return-value]
            boxes[keep], scores[keep], class_indices, task=self.model_info.tasks[0], meta=meta
        )


@register_flavor
class OpenVINOTorchvisionDetection(OpenVINOModel):
    """torchvision detection models exported as full graph, e.g. Faster R-CNN."""

    framework = "torchvision"
    decoder = "boxes_scores_labels"
    params_model = BoxesScoresLabelsParams
    params: BoxesScoresLabelsParams

    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        class_indices, keep = map_labels(outputs["labels"], self.params.label_to_class_idx)
        return bboxes_from_input_xyxy(  # type: ignore[return-value]
            outputs["boxes"][keep],
            outputs["scores"][keep],
            class_indices,
            task=self.model_info.tasks[0],
            meta=meta,
            score_threshold=self.params.score_threshold,
        )
