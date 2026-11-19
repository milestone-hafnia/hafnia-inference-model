"""Export RF-DETR detection models to ONNX model bundles.

Sources: a model archive of trainer-object-detection (e.g. a platform checkpoint) or pretrained COCO weights by
name. The network is exported with RF-DETR's own ONNX export: input '[N, 3, R, R]' with a dynamic batch axis,
outputs one box per query ('dets', normalized center x, center y, width, height) and the class logits per query
('labels'). Sigmoid, top-k over (query, class) and the box conversion run in the flavor (decoder
'rfdetr_detection'), as RF-DETR's 'PostProcess'. DETR-style models need no NMS.

Preprocessing follows RF-DETR's 'predict': resize to R x R (torchvision 'resize', antialiased on tensors), scale
to [0, 1], the model's mean/std.

Logit column i is class i of the task. Placeholder classes (COCO ids without a category, named 'NotDefined_*'
in the trainer's archives) and the extra "no object" column are not task classes: 'label_to_class_idx' maps
them to -1.
"""

import importlib.metadata
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import onnx
import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from rfdetr import detr
from rfdetr.util.coco_classes import COCO_CLASSES

from inference_model_core import (
    InputInfo,
    ModelBundleInfo,
    PostprocessingInfo,
    PreprocessingInfo,
    QuantizationHints,
    ResizeInfo,
    RuntimeInfo,
    TensorInfo,
    write_bundle,
)
from model_exporters.onnx_export import DEFAULT_OPSET_VERSION
from model_exporters.trainer_archive import open_trainer_archive

FRAMEWORK = "rfdetr"
DECODER_RFDETR_DETECTION = "rfdetr_detection"
MODEL_FILENAME = "model.onnx"
TASK_NAME = "object_detection"
# Score threshold of trainer-object-detection's inference ('InferenceConfig.threshold')
DEFAULT_SCORE_THRESHOLD = 0.05
PLACEHOLDER_CLASS = re.compile(r"NotDefined_\d+")

# Model name (as in trainer-object-detection's 'model_config.json') -> RF-DETR class. Detection models only.
MODEL_CLASSES: Dict[str, Type[detr.RFDETR]] = {
    "RFDETRNano": detr.RFDETRNano,
    "RFDETRSmall": detr.RFDETRSmall,
    "RFDETRMedium": detr.RFDETRMedium,
    "RFDETRLarge": detr.RFDETRLarge,
}


def load_pretrained_model(model_name: str) -> Tuple[detr.RFDETR, List[str]]:
    """Pretrained COCO model (weights are downloaded on first use). Returns (model, category name per logit
    column), with placeholders for COCO ids without a category."""
    model = _model_class(model_name)()
    n_columns = model.model_config.num_classes + 1
    categories = [COCO_CLASSES.get(column, f"NotDefined_{column:03d}") for column in range(n_columns)]
    return model, categories


def load_trainer_archive(path_archive: Union[str, Path]) -> Tuple[detr.RFDETR, List[str], str]:
    """Model from a trainer-object-detection archive. Returns (model, task class names, model name)."""
    with open_trainer_archive(path_archive) as (config, path_weights):
        model_name = config["name"]
        task = TaskInfo.model_validate(config["task"])
        if task.primitive is not Bbox:
            raise ValueError(f"Only detection (Bbox) models are supported, '{model_name}' is '{task.primitive}'.")
        model = _model_class(model_name)(pretrain_weights=str(path_weights))
    return model, list(task.get_class_names() or []), model_name


def export_pretrained_rfdetr_detection(
    model_name: str,
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    model, categories = load_pretrained_model(model_name)
    return export_rfdetr_detection(
        model,
        categories,
        path_bundle,
        name=model_name,
        variant=model_name,
        opset_version=opset_version,
        overwrite=overwrite,
        provenance={"weights": Path(model.model_config.pretrain_weights or "").name},
    )


def export_trainer_archive_rfdetr(
    path_archive: Union[str, Path],
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    model, categories, model_name = load_trainer_archive(path_archive)
    return export_rfdetr_detection(
        model,
        categories,
        path_bundle,
        name=model_name,
        variant=model_name,
        opset_version=opset_version,
        overwrite=overwrite,
        provenance={"trainer_archive": Path(path_archive).name},
    )


def export_rfdetr_detection(
    model: detr.RFDETR,
    categories: List[str],
    path_bundle: Union[str, Path],
    name: str,
    variant: Optional[str] = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
    provenance: Optional[dict] = None,
) -> Path:
    """Export an RF-DETR detection model as an ONNX model bundle.

    Args:
        model: RF-DETR model ('rfdetr.detr.RFDETR'). Moved to CPU by the export.
        categories: Category name per logit column (column i = class i). Placeholders ('NotDefined_*') are not
            task classes. Columns beyond the list (the "no object" column) are ignored.
        path_bundle: Output bundle, a '.zip' path or a directory.
        name: Model name stored in the model info.
        variant: Informational variant stored in the model info.
        opset_version: ONNX opset.
        overwrite: Replace an existing bundle.
        provenance: Extra provenance information stored in the model info.
    """
    model_config = model.model_config
    if model_config.segmentation_head or getattr(model_config, "use_grouppose_keypoints", False):
        raise ValueError("Only RF-DETR detection models are supported (no segmentation or keypoint heads).")
    resolution = model.model.resolution
    n_queries, n_columns = model_config.num_queries, model_config.num_classes + 1
    task, label_to_class_idx = task_from_categories(categories, n_columns)

    with tempfile.TemporaryDirectory(prefix="rfdetr_export_") as tmp_dir:
        model.export(output_dir=tmp_dir, dynamic_batch=True, opset_version=opset_version, verbose=False)
        (path_onnx,) = Path(tmp_dir).glob("*.onnx")
        onnx.checker.check_model(path_onnx.as_posix())

        runtime = RuntimeInfo(
            model_format="onnx",
            inference_engine="onnxruntime",
            file_path=MODEL_FILENAME,
            precision="fp32",
            input=InputInfo(tensors=[TensorInfo(name="input", shape=[None, 3, resolution, resolution])]),
            outputs=[
                TensorInfo(name="dets", shape=[None, n_queries, 4]),
                TensorInfo(name="labels", shape=[None, n_queries, n_columns]),
            ],
            preprocessing=PreprocessingInfo(
                color_order="RGB",
                resize=ResizeInfo(
                    mode="stretch", height=resolution, width=resolution, interpolation="linear", antialias=True
                ),
                scale=1 / 255,
                mean=list(model.means),
                std=list(model.stds),
                layout="NCHW",
                dtype="float32",
            ),
            # NNCF's default quantized the transformer as a CNN and dropped mAP from 0.66 to 0.0 on the micro
            # dataset. Its transformer mode keeps softmax, layer norm and similar operations in floating point.
            quantization_hints=QuantizationHints(model_type="transformer"),
            postprocessing=PostprocessingInfo(
                decoder=DECODER_RFDETR_DETECTION,
                params={
                    "score_threshold": DEFAULT_SCORE_THRESHOLD,
                    "num_select": model.model.postprocess.num_select,
                    "label_to_class_idx": label_to_class_idx,
                },
            ),
            provenance={
                "exporter": __name__,
                "rfdetr": importlib.metadata.version("rfdetr"),
                "torch": torch.__version__,
                "opset_version": opset_version,
                **(provenance or {}),
            },
        )
        model_info = ModelBundleInfo(name=name, framework=FRAMEWORK, variant=variant, tasks=[task], runtime=runtime)
        return write_bundle(path_bundle, model_info, files={MODEL_FILENAME: path_onnx}, overwrite=overwrite)


def task_from_categories(categories: List[str], n_columns: int) -> Tuple[TaskInfo, List[int]]:
    """Task with the real classes, and the mapping from logit column to task class index (-1 = ignored)."""
    if len(categories) > n_columns:
        raise ValueError(f"Got {len(categories)} categories for a model with {n_columns} logit columns.")
    class_names: List[str] = []
    label_to_class_idx = [-1] * n_columns
    for column, category in enumerate(categories):
        if PLACEHOLDER_CLASS.fullmatch(category):
            continue
        label_to_class_idx[column] = len(class_names)
        class_names.append(category)
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=class_names, name=TASK_NAME)
    return task, label_to_class_idx


def _model_class(model_name: str) -> Type[detr.RFDETR]:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported RF-DETR model '{model_name}'. Options: {list(MODEL_CLASSES)}")
    return MODEL_CLASSES[model_name]
