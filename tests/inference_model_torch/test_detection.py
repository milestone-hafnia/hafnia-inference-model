import pytest
import torch
from torchvision import ops as torchvision_ops

from inference_model_torch.components.detection import (
    select_anchor_detections,
    select_query_detections,
    task_label_mask,
)


def random_boxes(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    top_left = torch.rand(n, 2, generator=generator) * 300
    return torch.cat([top_left, top_left + torch.rand(n, 2, generator=generator) * 60 + 1], dim=1)


def test_select_anchor_detections_matches_torchvision_ssd_logic():
    """Reference: the per-class loop of 'torchvision.models.detection.SSD.postprocess_detections'."""
    n_anchors, n_classes = 1000, 6
    boxes = random_boxes(n_anchors, seed=3)
    scores = torch.softmax(torch.randn(n_anchors, n_classes, generator=torch.Generator().manual_seed(4)) * 3, dim=-1)
    score_threshold, nms_iou, topk, max_detections = 0.2, 0.55, 50, 40

    ref_boxes, ref_scores, ref_labels = [], [], []
    for label in range(1, n_classes):
        label_scores = scores[:, label]
        keep = label_scores > score_threshold
        label_scores, idx = label_scores[keep].topk(min(topk, int(keep.sum())))
        ref_boxes.append(boxes[keep][idx])
        ref_scores.append(label_scores)
        ref_labels.append(torch.full_like(idx, label))
    ref_boxes_t, ref_scores_t, ref_labels_t = torch.cat(ref_boxes), torch.cat(ref_scores), torch.cat(ref_labels)
    keep = torchvision_ops.batched_nms(ref_boxes_t, ref_scores_t, ref_labels_t, nms_iou)[:max_detections]

    out_boxes, out_scores, out_labels = select_anchor_detections(
        boxes, scores, score_threshold, nms_iou, topk, max_detections, background_index=0
    )

    assert len(out_scores) == len(keep) == max_detections
    assert torch.equal(out_scores, ref_scores_t[keep])
    assert torch.equal(out_boxes, ref_boxes_t[keep])
    assert torch.equal(out_labels, ref_labels_t[keep])


def test_select_anchor_detections_without_background_class():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.7], [0.2, 0.6]])
    _, out_scores, out_labels = select_anchor_detections(boxes, scores, 0.5, 0.5, 10, 10, background_index=None)
    # Class 0: the second box overlaps the first and is suppressed. Class 1: both boxes are kept
    assert out_scores.tolist() == pytest.approx([0.9, 0.7, 0.6])
    assert out_labels.tolist() == [0, 1, 1]


def test_select_anchor_detections_without_candidates():
    boxes, scores, labels = select_anchor_detections(random_boxes(10), torch.full((10, 3), 0.01), 0.5, 0.5, 10, 10)
    assert boxes.shape == (0, 4) and scores.shape == (0,) and labels.shape == (0,)


def test_task_label_mask():
    assert task_label_mask(torch.tensor([0, 2, 3, 1]), n_classes=3).tolist() == [True, True, False, True]
    assert task_label_mask(torch.zeros(0, dtype=torch.int64), n_classes=3).tolist() == []
    with pytest.raises(ValueError, match=">= 0"):
        task_label_mask(torch.tensor([-1]), n_classes=3)


def test_select_query_detections_matches_rfdetr_postprocess():
    """Same scores, labels and boxes as RF-DETR's 'PostProcess' with the model input size as target size."""
    postprocess_module = pytest.importorskip("rfdetr.models.postprocess")
    generator = torch.Generator().manual_seed(0)
    n_queries, n_columns, input_hw = 50, 7, (384, 512)
    logits = torch.randn(n_queries, n_columns, generator=generator)
    boxes = torch.cat(
        [
            torch.rand(n_queries, 2, generator=generator) * 0.6 + 0.2,
            torch.rand(n_queries, 2, generator=generator) * 0.25 + 0.05,
        ],
        dim=1,
    )

    expected = postprocess_module.PostProcess(num_select=30)(
        {"pred_logits": logits[None], "pred_boxes": boxes[None]}, target_sizes=torch.tensor([input_hw])
    )[0]
    boxes_xyxy, scores, labels = select_query_detections(boxes, logits, num_select=30, input_hw=input_hw)

    assert torch.allclose(scores, expected["scores"], atol=1e-6)
    assert torch.equal(labels, expected["labels"])
    assert torch.allclose(boxes_xyxy, expected["boxes"], atol=1e-3)
