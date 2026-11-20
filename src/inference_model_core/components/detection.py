"""Detection components: decoders that select detections, and conversion to 'Bbox' primitives."""

from typing import NamedTuple, Optional

import numpy as np
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from pydantic import Field, PrivateAttr

from inference_model_core.component import NumpyComponent, register_component
from inference_model_core.components.outputs import Predictions
from inference_model_core.detection import select_anchor_detections, select_query_detections, task_label_mask
from inference_model_core.preprocessing import ImageTransform
from inference_model_core.primitives import bboxes_from_input_xyxy


class Detections(NamedTuple):
    boxes: np.ndarray  # (N, 4) xyxy pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,)


@register_component
class AnchorNMS(NumpyComponent):
    """Dense per-anchor predictions -> detections: score threshold, top-k and NMS per class.

    Inputs 'boxes' (A, 4) decoded xyxy boxes and 'scores' (A, C) class scores. Output 'labels' are score columns.
    """

    type = "anchor_nms"

    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    topk_candidates: int = Field(gt=0, description="Highest scoring candidates kept per class before NMS")
    max_detections: int = Field(gt=0)
    background_index: Optional[int] = Field(description="Score column of the background class, if any")

    def forward(self, boxes: np.ndarray, scores: np.ndarray) -> Detections:
        detections = select_anchor_detections(
            boxes=np.asarray(boxes),
            scores=np.asarray(scores),
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            topk_candidates=self.topk_candidates,
            max_detections=self.max_detections,
            background_index=self.background_index,
        )
        return Detections(*detections)


@register_component
class QueryTopK(NumpyComponent):
    """Detections of a DETR-style model: sigmoid class scores and top-k over (query, class). No NMS.

    Inputs 'boxes' (Q, 4) normalized (center x, center y, width, height), 'logits' (Q, C) and the 'transform' of
    the resize. Output 'boxes' are xyxy in network input pixels, 'labels' are logit columns.
    """

    type = "query_topk"

    num_select: int = Field(gt=0, description="Highest scoring (query, class) pairs kept")

    def forward(self, boxes: np.ndarray, logits: np.ndarray, transform: ImageTransform) -> Detections:
        detections = select_query_detections(
            boxes_cxcywh=np.asarray(boxes),
            logits=np.asarray(logits),
            num_select=self.num_select,
            input_hw=(transform.input_height, transform.input_width),
        )
        return Detections(*detections)


@register_component
class DetectionsToBboxes(NumpyComponent):
    """Detections -> 'Bbox' predictions on the original image.

    Inputs 'boxes' (N, 4) xyxy pixels, 'scores' (N,) and 'labels' (N,), where label i is class i of the task.
    Labels beyond the task's classes are dropped (e.g. RF-DETR's "no object" column). The boxes are in pixels of
    the network input when 'transform' is connected, or of the original 'image' when that is connected instead.
    """

    type = "detections_to_bboxes"

    task_name: str = Field(description="Task of the model info the detections belong to")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Detections below it are dropped")

    _task: TaskInfo = PrivateAttr()

    def setup(self) -> None:
        if ("transform" in self.info.inputs) == ("image" in self.info.inputs):
            raise ValueError(f"Component '{self.info.name}' needs either the 'transform' or the 'image' input.")
        self._task = self.context.model_info.get_task_by_name(self.task_name)

    def forward(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        transform: Optional[ImageTransform] = None,
        image: Optional[np.ndarray] = None,
    ) -> Predictions:
        if transform is None:
            if image is None:
                raise ValueError("Either 'transform' or 'image' is needed to map the boxes to the original image.")
            transform = ImageTransform.identity(*image.shape[:2])
        boxes, scores, labels = np.asarray(boxes), np.asarray(scores), np.asarray(labels)
        keep = task_label_mask(labels, len(self._task.get_class_names() or []))
        predictions = bboxes_from_input_xyxy(
            boxes[keep],
            scores[keep],
            labels[keep],
            task=self._task,
            transform=transform,
            score_threshold=self.score_threshold,
        )
        return Predictions(predictions)  # type: ignore[arg-type]
