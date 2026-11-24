import pytest
import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification

from inference_model_torch.components.classification import top1_classification

TASK = TaskInfo.from_class_names(primitive=Classification, class_names=["cat", "dog", "car"], name="classification")


def test_top1_classification():
    (prediction,) = top1_classification(torch.tensor([0.1, 0.7, 0.2]), TASK)
    assert (prediction.class_idx, prediction.class_name, prediction.task_name) == (1, "dog", "classification")
    assert prediction.confidence == pytest.approx(0.7)
    assert not prediction.ground_truth
    assert top1_classification(torch.tensor([0.4, 0.3, 0.3]), TASK, score_threshold=0.5) == []
    with pytest.raises(ValueError, match="2 class probabilities"):
        top1_classification(torch.tensor([0.5, 0.5]), TASK)
