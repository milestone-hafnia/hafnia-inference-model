import pytest
from helpers import write_dummy_bundle

from inference_model_core import GraphInferenceModel


@pytest.fixture
def dummy_model(tmp_path, dummy_network):
    """Detects label 0 and label 1 on every image, as the classes 'class_names[0]' and '[1]'."""

    def load(class_names) -> GraphInferenceModel:
        return GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "model.zip", class_names=class_names))

    return load


def test_prediction_classes_are_aligned_to_the_ground_truth(micro_coco, dummy_model):
    """A model with more classes than the dataset (e.g. placeholders): those classes are removed for the metrics."""
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()
    model = dummy_model(["NotDefined_000", *coco_class_names])  # Label 0 is the placeholder, label 1 'person'

    metrics, predictions = model.benchmark(micro_coco)
    prediction_task = predictions.info.get_task_by_name("object_detection/predictions")
    assert prediction_task.get_class_names() == coco_class_names
    predicted_names = {box["class_name"] for row in predictions.samples["bboxes"].to_list() for box in row if box}
    assert "NotDefined_000" not in predicted_names
    assert "person" in predicted_names
    assert metrics


def test_different_label_spaces_need_a_class_mapping(micro_coco, dummy_model):
    with pytest.raises(ValueError, match="label spaces differ"):
        dummy_model(["car", "person"]).benchmark(micro_coco)
