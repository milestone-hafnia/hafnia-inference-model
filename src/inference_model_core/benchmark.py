"""The recipe transform that aligns the prediction classes to the ground truth by name, for Hafnia's benchmark.

Hafnia's metrics need a prediction task to have exactly the class names of its ground-truth task, in the same
order. A model's classes often differ from the dataset's even when the model predicts every class of the
dataset: it can predict additional classes, or order the classes differently. E.g. COCO models have 91 classes
including 'NotDefined_*' placeholders, while 'coco-2017' has 80.
"""

from typing import Callable, List, Optional, Tuple

from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset


class AlignPredictionClasses(RecipeTransform):
    """Align the classes of prediction tasks to those of their ground-truth tasks, matching classes by name.

    Aligning only works when the model predicts every class of the ground truth. It then handles the two ways the
    classes can still differ: it removes additional classes, which the ground truth doesn't have, together with
    their predictions, and it reorders the classes to the order of the ground truth. A model that doesn't predict
    every class of the ground truth raises an error: map its classes to the ground truth's with a 'ClassMapper'
    first.

    This is useful for benchmarking models pretrained on COCO. They "predict" 91 classes, of which only 80 are
    used, while our version of the COCO dataset only has the 80 classes that have annotations. Aligning removes the
    unused classes and puts the remaining ones in the order of the dataset.

    Use it as a recipe transform in Hafnia's 'run_benchmark' to align the predictions of a model to the ground
    truth of a dataset:

    Example:
        >>> recipe = AlignPredictionClasses()
        >>> metrics, _ = run_benchmark(micro_coco, model_pretrained, recipe_transforms=[recipe])

    Attributes:
        align_tasks: Pairs of (prediction task name, ground-truth task name) to align. Default: every prediction
            task named '<ground-truth task name>/predictions' (Hafnia's default postfix) whose ground-truth task is
            in the dataset. Prediction tasks whose classes already match are unchanged.
    """

    align_tasks: Optional[List[Tuple[str, str]]] = None

    @staticmethod
    def get_function() -> Callable[..., HafniaDataset]:
        return align_prediction_classes


def align_prediction_classes(
    dataset: HafniaDataset, align_tasks: Optional[List[Tuple[str, str]]] = None
) -> HafniaDataset:
    """The function of 'AlignPredictionClasses'."""
    if align_tasks is None:
        align_tasks = default_align_tasks(dataset)
    for prediction_task_name, task_name in align_tasks:
        ground_truth_classes = dataset.info.get_task_by_name(task_name).get_class_names() or []
        prediction_classes = dataset.info.get_task_by_name(prediction_task_name).get_class_names() or []
        if not ground_truth_classes or prediction_classes == ground_truth_classes:
            continue

        missing = [name for name in ground_truth_classes if name not in prediction_classes]
        if missing:
            raise ValueError(
                f"The prediction task '{prediction_task_name}' has no classes {missing}, which the ground-truth task "
                f"'{task_name}' has. The label spaces differ: map the classes of the predictions and/or the ground "
                "truth to a common one first, e.g. with a 'ClassMapper'."
            )
        dataset = dataset.class_mapper(
            {name: name for name in ground_truth_classes}, method="remove_undefined", task_name=prediction_task_name
        )
    return dataset


def default_align_tasks(dataset: HafniaDataset) -> List[Tuple[str, str]]:
    """(prediction task name, ground-truth task name) of each task named '<ground-truth task name>/predictions'
    whose ground-truth task is in 'dataset'."""
    task_names = [task.name for task in dataset.info.tasks if task.name is not None]
    align_tasks = []
    for prediction_task_name in task_names:
        task_name = prediction_task_name.removesuffix(TASK_NAME_PREDICTIONS_POSTFIX)
        if task_name != prediction_task_name and task_name in task_names:
            align_tasks.append((prediction_task_name, task_name))
    return align_tasks
