"""The recipe that aligns the prediction classes to the ground truth by name, for Hafnia's benchmark.

Hafnia's metrics need the prediction task to have exactly the class names of the ground-truth task, in the same
order. Bundles have one class per model label, e.g. 91 for COCO models including 'NotDefined_*' placeholders,
while datasets often have fewer (80 for 'coco-2017'). Aligning keeps the dataset's classes, in the dataset's
order, and removes predictions of other classes.
"""

from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.dataset_recipe.recipe_transforms import ClassMapper
from hafnia.dataset.hafnia_dataset import HafniaDataset


def align_classes_recipe(
    dataset: HafniaDataset, task_name: str, prediction_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX
) -> ClassMapper:
    """Recipe transform that gives the predictions of 'task_name' the class names of the ground truth in 'dataset'.

    Pass it to Hafnia's 'run_benchmark' as the last of the 'recipe_transforms'. It raises when the predictions
    lack a class of the ground truth: then the label spaces differ, and the predictions and/or the ground truth
    need a class mapping to a common one first.
    """
    ground_truth_classes = dataset.info.get_task_by_name(task_name).get_class_names() or []
    return ClassMapper(
        class_mapping={name: name for name in ground_truth_classes},
        method="remove_undefined",
        task_name=f"{task_name}{prediction_postfix}",
    )
