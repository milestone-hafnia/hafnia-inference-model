import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark
from helpers import write_dummy_bundle

from hafnia_model_bundle.core import AlignPredictionClasses, GraphInferenceModel


@pytest.fixture
def dummy_model(tmp_path, dummy_network):
    """Detects label 0 and label 1 on every image, as the classes 'class_names[0]' and '[1]'."""

    def load(class_names) -> GraphInferenceModel:
        return GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "model.zip", class_names=class_names))

    return load


def predicted_class_names(predictions) -> set:
    boxes = [box for row in predictions.samples["bboxes"].to_list() for box in row if box]
    return {box["class_name"] for box in boxes if box["task_name"] == "object_detection/predictions"}


def test_additional_prediction_classes_are_removed(micro_coco, dummy_model):
    """A model with more classes than the dataset (e.g. placeholders): those classes are removed for the metrics."""
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()
    model = dummy_model(["NotDefined_000", *coco_class_names])  # Label 0 is the placeholder, label 1 'person'

    metrics, predictions = run_benchmark(micro_coco, model, recipe_transforms=[AlignPredictionClasses()])
    prediction_task = predictions.info.get_task_by_name("object_detection/predictions")
    assert prediction_task.get_class_names() == coco_class_names
    assert "NotDefined_000" not in predicted_class_names(predictions)
    assert "person" in predicted_class_names(predictions)
    assert metrics


def test_prediction_classes_are_put_in_the_ground_truth_order(micro_coco, dummy_model):
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()
    model = dummy_model(list(reversed(coco_class_names)))

    metrics, predictions = run_benchmark(micro_coco, model, recipe_transforms=[AlignPredictionClasses()])
    assert predictions.info.get_task_by_name("object_detection/predictions").get_class_names() == coco_class_names
    assert predicted_class_names(predictions) == set(coco_class_names[-2:])
    assert metrics


def test_different_label_spaces_need_a_class_mapping(micro_coco, dummy_model):
    with pytest.raises(ValueError, match="label spaces differ"):
        run_benchmark(micro_coco, dummy_model(["car", "person"]), recipe_transforms=[AlignPredictionClasses()])


def test_explicit_align_tasks(micro_coco, dummy_model):
    """Prediction tasks with another name than '<task>/predictions' are aligned when named in 'align_tasks'."""
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()
    model = dummy_model(["NotDefined_000", *coco_class_names])
    postfix = "/model_a"

    with pytest.raises(ValueError, match="do not match"):  # Not aligned by default
        run_benchmark(micro_coco, model, postfix, recipe_transforms=[AlignPredictionClasses()])
    recipe = AlignPredictionClasses(align_tasks=[("object_detection/model_a", "object_detection")])
    metrics, predictions = run_benchmark(micro_coco, model, postfix, recipe_transforms=[recipe])
    assert predictions.info.get_task_by_name("object_detection/model_a").get_class_names() == coco_class_names
    assert metrics
