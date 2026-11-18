import numpy as np
import pytest

from inference_model_core.detection import batched_nms, map_labels, nms, select_anchor_detections

torch = pytest.importorskip("torch")
torchvision_ops = pytest.importorskip("torchvision.ops")


def random_boxes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    top_left = rng.random((n, 2)) * 300
    return np.concatenate([top_left, top_left + rng.random((n, 2)) * 60 + 1], axis=1).astype(np.float32)


@pytest.mark.parametrize("n_boxes", [0, 1, 50, 500])
def test_nms_matches_torchvision(n_boxes):
    boxes = random_boxes(n_boxes)
    scores = np.random.default_rng(1).random(n_boxes).astype(np.float32)

    expected = torchvision_ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.5).numpy()
    np.testing.assert_array_equal(nms(boxes, scores, 0.5), expected)


def test_batched_nms_matches_torchvision():
    boxes = random_boxes(800)
    rng = np.random.default_rng(2)
    scores = rng.random(800).astype(np.float32)
    labels = rng.integers(0, 5, 800)

    expected = torchvision_ops.batched_nms(
        torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(labels), 0.55
    ).numpy()
    np.testing.assert_array_equal(batched_nms(boxes, scores, labels, 0.55), expected)


def test_select_anchor_detections_matches_torchvision_ssd_logic():
    """Reference: the per-class loop of 'torchvision.models.detection.SSD.postprocess_detections'."""
    n_anchors, n_classes = 1000, 6
    boxes = random_boxes(n_anchors, seed=3)
    logits = np.random.default_rng(4).normal(size=(n_anchors, n_classes)).astype(np.float32) * 3
    scores = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    score_threshold, nms_iou, topk, max_detections = 0.2, 0.55, 50, 40

    ref_boxes, ref_scores, ref_labels = [], [], []
    for label in range(1, n_classes):
        label_scores = torch.from_numpy(scores[:, label])
        keep = label_scores > score_threshold
        label_scores, idx = label_scores[keep].topk(min(topk, int(keep.sum())))
        ref_boxes.append(torch.from_numpy(boxes)[keep][idx])
        ref_scores.append(label_scores)
        ref_labels.append(torch.full_like(idx, label))
    ref_boxes_t, ref_scores_t, ref_labels_t = torch.cat(ref_boxes), torch.cat(ref_scores), torch.cat(ref_labels)
    keep = torchvision_ops.batched_nms(ref_boxes_t, ref_scores_t, ref_labels_t, nms_iou)[:max_detections]

    out_boxes, out_scores, out_labels = select_anchor_detections(
        boxes, scores, score_threshold, nms_iou, topk, max_detections, background_index=0
    )

    assert len(out_scores) == len(keep) == max_detections
    np.testing.assert_allclose(out_scores, ref_scores_t[keep].numpy())
    np.testing.assert_allclose(out_boxes, ref_boxes_t[keep].numpy())
    np.testing.assert_array_equal(out_labels, ref_labels_t[keep].numpy())


def test_select_anchor_detections_without_candidates():
    boxes, scores, labels = select_anchor_detections(
        random_boxes(10), np.full((10, 3), 0.01, dtype=np.float32), 0.5, 0.5, 10, 10
    )
    assert boxes.shape == (0, 4) and scores.shape == (0,) and labels.shape == (0,)


def test_map_labels():
    class_indices, keep = map_labels(np.array([0, 1, 2, 3, 1]), label_to_class_idx=[-1, 0, -1, 1])
    np.testing.assert_array_equal(class_indices, [0, 1, 0])
    np.testing.assert_array_equal(keep, [False, True, False, True, True])

    with pytest.raises(ValueError, match="must be in"):
        map_labels(np.array([4]), label_to_class_idx=[-1, 0, -1, 1])
