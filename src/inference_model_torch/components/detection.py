"""Detection components in torch: decoders that select detections, and conversion to 'Bbox' primitives.

The decoders reproduce the source frameworks' semantics ('SSD.postprocess_detections', RF-DETR's 'PostProcess'),
so thresholding and NMS can run outside the network.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from torchvision.ops import batched_nms

from inference_model_core.definitions.detection import AnchorNMS, DetectionsToBboxes, QueryTopK
from inference_model_core.transform import ImageTransform
from inference_model_torch.component import Component, register_component

Detections = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # boxes (K, 4), scores (K,), labels (K,)


@register_component
class TorchAnchorNMS(Component[AnchorNMS]):
    def forward(self, boxes: torch.Tensor, scores: torch.Tensor) -> Dict[str, Any]:
        params: AnchorNMS = self.params
        return _detections(
            select_anchor_detections(
                boxes=boxes,
                scores=scores,
                score_threshold=params.score_threshold,
                nms_iou_threshold=params.nms_iou_threshold,
                topk_candidates=params.topk_candidates,
                max_detections=params.max_detections,
                background_index=params.background_index,
            )
        )


@register_component
class TorchQueryTopK(Component[QueryTopK]):
    def forward(self, boxes: torch.Tensor, logits: torch.Tensor, transform: ImageTransform) -> Dict[str, Any]:
        params: QueryTopK = self.params
        return _detections(
            select_query_detections(
                boxes_cxcywh=boxes,
                logits=logits,
                num_select=params.num_select,
                input_hw=(transform.input_height, transform.input_width),
            )
        )


@register_component
class TorchDetectionsToBboxes(Component[DetectionsToBboxes]):
    def setup(self) -> None:
        self.task = self.context.bundle.get_task_by_name(self.params.task_name)

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[ImageTransform] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if transform is None:
            if image is None:
                raise ValueError("Either 'transform' or 'image' is needed to map the boxes to the original image.")
            transform = ImageTransform.identity(*image.shape[:2])
        keep = task_label_mask(labels, len(self.task.get_class_names() or []))
        predictions = bboxes_from_input_xyxy(
            boxes[keep],
            scores[keep],
            labels[keep],
            task=self.task,
            transform=transform,
            score_threshold=self.params.score_threshold,
        )
        return {"predictions": predictions}


def select_anchor_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float,
    nms_iou_threshold: float,
    topk_candidates: int,
    max_detections: int,
    background_index: Optional[int] = 0,
) -> Detections:
    """Turn dense per-anchor predictions into final detections for one image.

    Per class (except the background): keep scores above 'score_threshold', take the 'topk_candidates'
    highest, then run NMS per class and keep the 'max_detections' highest scores overall.

    Args:
        boxes: (A, 4) decoded xyxy boxes per anchor.
        scores: (A, C) class scores per anchor.

    Returns:
        (boxes (K, 4), scores (K,), labels (K,)) sorted by decreasing score. Labels index the score columns.
    """
    n_anchors, n_classes = scores.shape
    top_scores, top_anchors = scores.topk(min(topk_candidates, n_anchors), dim=0)  # (k, C): candidates per class
    candidates = top_scores > score_threshold
    if background_index is not None:
        candidates[:, background_index] = False
    labels = torch.arange(n_classes, device=scores.device).expand_as(top_scores)

    candidate_boxes = boxes[top_anchors[candidates]]
    candidate_scores, candidate_labels = top_scores[candidates], labels[candidates]
    # Suppresses per label and returns the kept indices by decreasing score
    keep = batched_nms(candidate_boxes, candidate_scores, candidate_labels, nms_iou_threshold)[:max_detections]
    return candidate_boxes[keep], candidate_scores[keep], candidate_labels[keep]


def select_query_detections(
    boxes_cxcywh: torch.Tensor,
    logits: torch.Tensor,
    num_select: int,
    input_hw: Tuple[int, int],
) -> Detections:
    """Detections of a DETR-style model (one box per query, sigmoid class scores), as RF-DETR's 'PostProcess'.

    Takes the 'num_select' highest (query, class) scores over all queries and classes, so one query can give
    detections for several classes. No NMS.

    Args:
        boxes_cxcywh: (Q, 4) boxes as (center x, center y, width, height), normalized to [0, 1].
        logits: (Q, C) class logits.
        num_select: Number of detections to keep.
        input_hw: (height, width) of the model input, to scale boxes to input pixels.

    Returns:
        (boxes (K, 4) xyxy in model input pixels, scores (K,), labels (K,) as logit columns), by descending score.
    """
    scores_flat = logits.to(torch.float32).sigmoid().reshape(-1)
    scores, top = scores_flat.topk(min(num_select, scores_flat.numel()))
    queries = torch.div(top, logits.shape[1], rounding_mode="floor")
    labels = top % logits.shape[1]

    center_x, center_y, width, height = boxes_cxcywh[queries].to(torch.float32).unbind(dim=1)
    input_height, input_width = input_hw
    boxes = torch.stack(
        [
            (center_x - width / 2) * input_width,
            (center_y - height / 2) * input_height,
            (center_x + width / 2) * input_width,
            (center_y + height / 2) * input_height,
        ],
        dim=1,
    )
    return boxes, scores, labels


def task_label_mask(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Which labels are classes of the task. Bundles have one task class per model label, but a model can have
    extra outputs that are not classes, such as RF-DETR's "no object" logit column after the last class."""
    if labels.numel() and int(labels.min()) < 0:
        raise ValueError(f"Model labels must be >= 0, got {int(labels.min())}.")
    return labels < n_classes


def input_xyxy_to_normalized(boxes_xyxy: torch.Tensor, transform: ImageTransform) -> torch.Tensor:
    """Map (N, 4) pixel boxes in model input space to normalized xyxy boxes on the original image, clipped to
    [0, 1]. Computed in double precision on the CPU."""
    boxes = boxes_xyxy.detach().cpu().to(torch.float64).reshape(-1, 4)
    x = (boxes[:, [0, 2]] - transform.pad_left) / transform.scale_x / transform.original_width
    y = (boxes[:, [1, 3]] - transform.pad_top) / transform.scale_y / transform.original_height
    return torch.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], dim=1).clamp(0.0, 1.0)


def bboxes_from_input_xyxy(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    class_indices: torch.Tensor,
    task: TaskInfo,
    transform: ImageTransform,
    score_threshold: float = 0.0,
) -> List[Bbox]:
    """Convert decoded detections to 'Bbox' predictions on the original image.

    Args:
        boxes_xyxy: (N, 4) boxes in pixel coordinates of the model input.
        scores: (N,) confidence scores.
        class_indices: (N,) indices into 'task.classes'.
        task: The model task the boxes belong to.
        transform: How the original image maps to the model input.
        score_threshold: Detections with a lower score are dropped.
    """
    class_names = task.get_class_names()
    if class_names is None:
        raise ValueError(f"Task '{task.name}' has no classes defined.")

    boxes = input_xyxy_to_normalized(boxes_xyxy, transform).tolist()
    scores_list = scores.reshape(-1).tolist()
    class_indices_list = class_indices.reshape(-1).tolist()
    if not (len(boxes) == len(scores_list) == len(class_indices_list)):
        raise ValueError(
            f"Got {len(boxes)} boxes, {len(scores_list)} scores and {len(class_indices_list)} class indices. Lengths "
            "must match."
        )

    bboxes = []
    for (x0, y0, x1, y1), score, class_idx in zip(boxes, scores_list, class_indices_list, strict=True):
        if score < score_threshold or x1 <= x0 or y1 <= y0:
            continue
        class_idx = int(class_idx)
        if not 0 <= class_idx < len(class_names):
            raise ValueError(f"Class index {class_idx} is out of range for task '{task.name}' ({len(class_names)}).")
        bboxes.append(
            Bbox(
                top_left_x=x0,
                top_left_y=y0,
                width=x1 - x0,
                height=y1 - y0,
                class_idx=class_idx,
                class_name=class_names[class_idx],
                confidence=float(score),
                ground_truth=False,
                task_name=task.name,
            )
        )
    return bboxes


def _detections(detections: Detections) -> Dict[str, Any]:
    boxes, scores, labels = detections
    return {"boxes": boxes, "scores": scores, "labels": labels}
