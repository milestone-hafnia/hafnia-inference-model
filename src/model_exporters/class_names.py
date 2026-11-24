"""Class names of exported models: one task class per model label, so labels need no mapping at inference.

COCO-trained models (torchvision, RF-DETR) output COCO category ids: label 0 and 10 ids have no category.
These labels are classes of the task too, under unique placeholder names: 'NotDefined_000', 'NotDefined_012',
... as in trainer-object-detection's model archives. Benchmarks against datasets without them align the
prediction classes by name (see 'align_classes_recipe').
"""

from typing import List

from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox


def placeholder_class_name(label: int) -> str:
    return f"NotDefined_{label:03d}"


def detection_task(class_names: List[str], task_name: str) -> TaskInfo:
    """'Bbox' task with one class per model label. Class names must be unique."""
    duplicates = sorted({name for name in class_names if class_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Class names must be unique, got duplicates {duplicates}. Use 'placeholder_class_name'.")
    return TaskInfo.from_class_names(primitive=Bbox, class_names=class_names, name=task_name)
