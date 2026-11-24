"""Hafnia's benchmark for model bundles, with the prediction classes aligned to the ground truth by name.

Hafnia's metrics need the prediction task to have exactly the class names of the ground-truth task, in the same
order. Bundles have one class per model label, e.g. 91 for COCO models including 'NotDefined_*' placeholders,
while datasets often have fewer (80 for 'coco-2017'). Aligning keeps the dataset's classes, in the dataset's
order, and removes predictions of other classes.

Imported by 'GraphInferenceModel.benchmark' when it is called, as Hafnia's metrics load their own dependencies.
"""

from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.hafnia_dataset import HafniaDataset


def align_prediction_classes(
    dataset: HafniaDataset, task_name: str, prediction_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX
) -> HafniaDataset:
    """Give the predictions of 'task_name' the ground truth's class names.

    Unchanged when the class names already match. Raises when the ground truth has classes the predictions don't:
    then the label spaces differ, and a class mapping is needed (e.g. 'scripts/benchmark.py --model-class-mapping'
    in package-openvino).
    """
    prediction_task_name = f"{task_name}{prediction_postfix}"
    ground_truth_classes = dataset.info.get_task_by_name(task_name).get_class_names() or []
    prediction_classes = dataset.info.get_task_by_name(prediction_task_name).get_class_names() or []
    if prediction_classes == ground_truth_classes:
        return dataset

    missing = [name for name in ground_truth_classes if name not in prediction_classes]
    if missing:
        raise ValueError(
            f"The predictions of '{task_name}' have no classes {missing}, which the ground truth has. The label "
            "spaces differ: map the classes of the predictions and/or the ground truth to a common one."
        )
    keep_ground_truth_classes = {name: name for name in ground_truth_classes}
    return dataset.class_mapper(keep_ground_truth_classes, method="remove_undefined", task_name=prediction_task_name)
