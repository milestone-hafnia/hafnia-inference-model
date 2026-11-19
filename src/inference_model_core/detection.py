"""Optional NumPy helpers for detection decoders: NMS, per-class candidate selection, query top-k and label
mapping.

They reproduce the source frameworks' semantics ('torchvision.ops.batched_nms', 'SSD.postprocess_detections',
RF-DETR's 'PostProcess'), so thresholding and NMS can run outside the exported graph. NumPy only: nothing here
may depend on torch.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def box_iou(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) xyxy boxes."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    top_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    union = areas[:, None] + areas[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns indices of kept boxes, sorted by decreasing score.

    A box is suppressed when its IoU with a higher scoring kept box is larger than 'iou_threshold'. Memory
    is quadratic in the number of boxes, so call it per class on top-k candidates.
    """
    order = np.argsort(-scores, kind="stable")
    iou = box_iou(boxes[order])
    suppressed = np.zeros(len(order), dtype=bool)
    keep: List[int] = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed[i + 1 :] |= iou[i, i + 1 :] > iou_threshold
    return order[keep]


def batched_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float) -> np.ndarray:
    """NMS applied independently per label. Returns kept indices sorted by decreasing score."""
    keep = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        keep.append(idx[nms(boxes[idx], scores[idx], iou_threshold)])
    if not keep:
        return np.zeros(0, dtype=np.int64)
    kept = np.concatenate(keep)
    return kept[np.argsort(-scores[kept], kind="stable")]


def select_anchor_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    nms_iou_threshold: float,
    topk_candidates: int,
    max_detections: int,
    background_index: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn dense per-anchor predictions into final detections for one image.

    Per class (except the background): keep scores above 'score_threshold', take the 'topk_candidates'
    highest, then run NMS per class and keep the 'max_detections' highest scores overall.

    Args:
        boxes: (A, 4) decoded xyxy boxes per anchor.
        scores: (A, C) class scores per anchor.

    Returns:
        (boxes (K, 4), scores (K,), labels (K,)) sorted by decreasing score. Labels index the score columns.
    """
    candidate_boxes, candidate_scores, candidate_labels = [], [], []
    for label in range(scores.shape[1]):
        if label == background_index:
            continue
        label_scores = scores[:, label]
        idx = np.flatnonzero(label_scores > score_threshold)
        if idx.size > topk_candidates:
            idx = idx[np.argpartition(-label_scores[idx], topk_candidates - 1)[:topk_candidates]]
        if idx.size == 0:
            continue
        keep = idx[nms(boxes[idx], label_scores[idx], nms_iou_threshold)]
        candidate_boxes.append(boxes[keep])
        candidate_scores.append(label_scores[keep])
        candidate_labels.append(np.full(keep.size, label, dtype=np.int64))

    if not candidate_scores:
        return np.zeros((0, 4), dtype=boxes.dtype), np.zeros(0, dtype=scores.dtype), np.zeros(0, dtype=np.int64)

    all_scores = np.concatenate(candidate_scores)
    order = np.argsort(-all_scores, kind="stable")[:max_detections]
    return np.concatenate(candidate_boxes)[order], all_scores[order], np.concatenate(candidate_labels)[order]


def map_labels(labels: np.ndarray, label_to_class_idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Map raw model labels to task class indices. Labels mapped to -1 (e.g. background or unused 'N/A'
    entries of torchvision's COCO categories) are dropped.

    Returns:
        (class indices of the kept detections, boolean mask of kept detections)
    """
    mapping = np.asarray(label_to_class_idx, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= len(mapping)):
        raise ValueError(f"Model labels must be in [0, {len(mapping)}), got [{labels.min()}, {labels.max()}].")
    class_indices = mapping[labels]
    keep = class_indices >= 0
    return class_indices[keep], keep


def sigmoid(values: np.ndarray) -> np.ndarray:
    # Written with exp(-|x|) so large logits of either sign don't overflow
    exp = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))


def select_query_detections(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    num_select: int,
    input_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    scores_flat = sigmoid(logits.astype(np.float32)).reshape(-1)
    num_select = min(num_select, scores_flat.size)
    top = np.argpartition(-scores_flat, num_select - 1)[:num_select]
    top = top[np.argsort(-scores_flat[top], kind="stable")]
    queries, labels = np.divmod(top, logits.shape[1])

    center_x, center_y, width, height = boxes_cxcywh[queries].astype(np.float32).T
    input_height, input_width = input_hw
    boxes = np.stack(
        [
            (center_x - width / 2) * input_width,
            (center_y - height / 2) * input_height,
            (center_x + width / 2) * input_width,
            (center_y + height / 2) * input_height,
        ],
        axis=1,
    )
    return boxes, scores_flat[top], labels
