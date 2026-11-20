"""ONNX Runtime flavors. Registered through the 'hafnia.inference_models' entry point (see pyproject.toml)."""

from typing import List

from hafnia.dataset.primitives import Primitive
from pydantic import BaseModel, ConfigDict, Field

from inference_model_core import PreprocessMeta, RawTensors, register_flavor
from inference_model_core.classification import softmax, top1_classification
from inference_model_core.detection import select_anchor_detections, select_query_detections, task_label_mask
from inference_model_core.primitives import bboxes_from_input_xyxy
from inference_model_onnxruntime.engine import OnnxRuntimeModel


class AnchorBoxesNMSParams(BaseModel):
    """Parameters of the 'anchor_boxes_nms' decoder: per-anchor boxes and class scores, NMS outside the graph."""

    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    topk_candidates: int = Field(gt=0, description="Highest scoring candidates kept per class before NMS")
    max_detections: int = Field(gt=0)
    background_index: int | None = Field(description="Score column of the background class, if any")


class BoxesScoresLabelsParams(BaseModel):
    """Parameters of the 'boxes_scores_labels' decoder: final detections from the graph."""

    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(ge=0.0, le=1.0)


class RFDETRDetectionParams(BaseModel):
    """Parameters of the 'rfdetr_detection' decoder: one box per query and sigmoid class logits, top-k outside the
    graph."""

    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(ge=0.0, le=1.0)
    num_select: int = Field(gt=0, description="Highest scoring (query, class) pairs kept")


class ClassificationLogitsParams(BaseModel):
    """Parameters of the 'classification_logits' decoder: one logit per class, softmax outside the graph."""

    model_config = ConfigDict(extra="forbid")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="No prediction below this probability")


@register_flavor
class OnnxRuntimeTorchvisionSSD(OnnxRuntimeModel):
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
        task = self.model_info.tasks[0]
        keep = task_label_mask(labels, len(task.get_class_names() or []))
        return bboxes_from_input_xyxy(  # type: ignore[return-value]
            boxes[keep], scores[keep], labels[keep], task=task, meta=meta
        )


@register_flavor
class OnnxRuntimeTorchvisionDetection(OnnxRuntimeModel):
    """torchvision detection models exported as full graph, e.g. Faster R-CNN."""

    framework = "torchvision"
    decoder = "boxes_scores_labels"
    params_model = BoxesScoresLabelsParams
    params: BoxesScoresLabelsParams

    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        task = self.model_info.tasks[0]
        keep = task_label_mask(outputs["labels"], len(task.get_class_names() or []))
        return bboxes_from_input_xyxy(  # type: ignore[return-value]
            outputs["boxes"][keep],
            outputs["scores"][keep],
            outputs["labels"][keep],
            task=task,
            meta=meta,
            score_threshold=self.params.score_threshold,
        )


@register_flavor
class OnnxRuntimeTorchvisionClassification(OnnxRuntimeModel):
    """torchvision classification models exported as network only ('model_exporters.torchvision_classification')."""

    framework = "torchvision"
    decoder = "classification_logits"
    params_model = ClassificationLogitsParams
    params: ClassificationLogitsParams

    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        probabilities = softmax(outputs["logits"][0])
        return top1_classification(  # type: ignore[return-value]
            probabilities, task=self.model_info.tasks[0], score_threshold=self.params.score_threshold
        )


@register_flavor
class OnnxRuntimeRFDETRDetection(OnnxRuntimeModel):
    """RF-DETR detection models ('model_exporters.rfdetr_detection')."""

    framework = "rfdetr"
    decoder = "rfdetr_detection"
    params_model = RFDETRDetectionParams
    params: RFDETRDetectionParams

    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        boxes, scores, labels = select_query_detections(
            boxes_cxcywh=outputs["dets"][0],
            logits=outputs["labels"][0],
            num_select=self.params.num_select,
            input_hw=(meta.input_height, meta.input_width),
        )
        task = self.model_info.tasks[0]
        keep = task_label_mask(labels, len(task.get_class_names() or []))
        return bboxes_from_input_xyxy(  # type: ignore[return-value]
            boxes[keep],
            scores[keep],
            labels[keep],
            task=task,
            meta=meta,
            score_threshold=self.params.score_threshold,
        )
