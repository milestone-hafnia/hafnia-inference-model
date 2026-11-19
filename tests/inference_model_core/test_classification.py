import numpy as np
import pytest
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification

from inference_model_core.classification import softmax, top1_classification

TASK = TaskInfo.from_class_names(primitive=Classification, class_names=["cat", "dog", "car"], name="classification")


def test_softmax():
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities[0], np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum())
    assert np.isfinite(probabilities).all()  # Large logits don't overflow


def test_top1_classification():
    (prediction,) = top1_classification(np.array([0.1, 0.7, 0.2]), TASK)
    assert (prediction.class_idx, prediction.class_name, prediction.task_name) == (1, "dog", "classification")
    assert prediction.confidence == pytest.approx(0.7)
    assert not prediction.ground_truth
    assert top1_classification(np.array([0.4, 0.3, 0.3]), TASK, score_threshold=0.5) == []
    with pytest.raises(ValueError, match="2 class probabilities"):
        top1_classification(np.array([0.5, 0.5]), TASK)
