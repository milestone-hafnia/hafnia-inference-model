"""Export torchvision classification models (ResNet18) to ONNX model bundles.

Sources: a model archive of trainer-classification (e.g. a platform checkpoint) or pretrained ImageNet weights by
name. The network is exported as is: input '[N, 3, S, S]' with a dynamic batch axis, output the class logits.
Softmax runs in the flavor (decoder 'classification_logits').

Preprocessing follows trainer-classification's training transform: resize to 'image_size' x 'image_size'
(torchvision 'Resize', antialiased on tensors), scale to [0, 1], ImageNet mean/std.
"""

import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import onnx
import torch
import torchvision
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification
from torch import nn
from torchvision.models import ResNet18_Weights, WeightsEnum, resnet18

from inference_model_core import (
    InputInfo,
    ModelBundleInfo,
    PostprocessingInfo,
    PreprocessingInfo,
    ResizeInfo,
    RuntimeInfo,
    TensorInfo,
    write_bundle,
)
from model_exporters.onnx_export import DEFAULT_OPSET_VERSION, onnx_export
from model_exporters.trainer_archive import open_trainer_archive

FRAMEWORK = "torchvision"
DECODER_CLASSIFICATION_LOGITS = "classification_logits"
MODEL_FILENAME = "model.onnx"
TASK_NAME = "classification"
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Architecture name (as in trainer-classification's 'model_config.json') -> (builder, ImageNet weights)
ARCHITECTURES: Dict[str, Tuple[Callable[..., nn.Module], WeightsEnum]] = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
}


def load_pretrained_model(model_name: str) -> Tuple[nn.Module, TaskInfo, int]:
    """Pretrained ImageNet model. Returns (model in eval mode, task with the ImageNet classes, image size)."""
    builder, weights = _architecture(model_name)
    model = builder(weights=weights).eval()
    task = TaskInfo.from_class_names(
        primitive=Classification, class_names=list(weights.meta["categories"]), name=TASK_NAME
    )
    return model, task, DEFAULT_IMAGE_SIZE


def load_trainer_archive(path_archive: Union[str, Path]) -> Tuple[nn.Module, TaskInfo, int, str]:
    """Model from a trainer-classification archive. Returns (model in eval mode, task, image size, model name)."""
    with open_trainer_archive(path_archive) as (config, path_weights):
        model_name = config["name"]
        task = TaskInfo.model_validate(config["task"])
        builder, _ = _architecture(model_name)
        model = builder(weights=None, num_classes=len(task.get_class_names() or []))
        model.load_state_dict(torch.load(path_weights, map_location="cpu", weights_only=True))
    return model.eval(), task, int(config.get("image_size", DEFAULT_IMAGE_SIZE)), model_name


def export_pretrained_torchvision_classification(
    model_name: str,
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    model, task, image_size = load_pretrained_model(model_name)
    return export_torchvision_classification(
        model,
        task,
        image_size,
        path_bundle,
        name=model_name,
        variant=model_name,
        opset_version=opset_version,
        overwrite=overwrite,
        provenance={"weights": str(ARCHITECTURES[model_name][1])},
    )


def export_trainer_archive_classification(
    path_archive: Union[str, Path],
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    model, task, image_size, model_name = load_trainer_archive(path_archive)
    return export_torchvision_classification(
        model,
        task,
        image_size,
        path_bundle,
        name=model_name,
        variant=model_name,
        opset_version=opset_version,
        overwrite=overwrite,
        provenance={"trainer_archive": Path(path_archive).name},
    )


def export_torchvision_classification(
    model: nn.Module,
    task: TaskInfo,
    image_size: int,
    path_bundle: Union[str, Path],
    name: str,
    variant: Optional[str] = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
    provenance: Optional[dict] = None,
) -> Path:
    """Export a torchvision classification model as an ONNX model bundle.

    Args:
        model: Classification model with one logit per class of 'task'. It is not modified.
        task: Classification task with the class names in logit order.
        image_size: Square input size the model was trained on.
        path_bundle: Output bundle, a '.zip' path or a directory.
        name: Model name stored in the model info.
        variant: Informational variant stored in the model info.
        opset_version: ONNX opset.
        overwrite: Replace an existing bundle.
        provenance: Extra provenance information stored in the model info.
    """
    n_classes = len(task.get_class_names() or [])
    with tempfile.TemporaryDirectory(prefix="classification_export_") as tmp_dir:
        path_onnx = Path(tmp_dir) / MODEL_FILENAME
        onnx_export(
            model,
            (torch.zeros(2, 3, image_size, image_size),),
            path_onnx,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset_version,
        )
        onnx.checker.check_model(path_onnx.as_posix())

        runtime = RuntimeInfo(
            model_format="onnx",
            inference_engine="onnxruntime",
            file_path=MODEL_FILENAME,
            precision="fp32",
            input=InputInfo(tensors=[TensorInfo(name="images", shape=[None, 3, image_size, image_size])]),
            outputs=[TensorInfo(name="logits", shape=[None, n_classes])],
            preprocessing=PreprocessingInfo(
                color_order="RGB",
                resize=ResizeInfo(
                    mode="stretch", height=image_size, width=image_size, interpolation="linear", antialias=True
                ),
                scale=1 / 255,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                layout="NCHW",
                dtype="float32",
            ),
            postprocessing=PostprocessingInfo(decoder=DECODER_CLASSIFICATION_LOGITS, params={"score_threshold": 0.0}),
            provenance={
                "exporter": __name__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "opset_version": opset_version,
                **(provenance or {}),
            },
        )
        model_info = ModelBundleInfo(name=name, framework=FRAMEWORK, variant=variant, tasks=[task], runtime=runtime)
        return write_bundle(path_bundle, model_info, files={MODEL_FILENAME: path_onnx}, overwrite=overwrite)


def _architecture(model_name: str) -> Tuple[Callable[..., nn.Module], WeightsEnum]:
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unsupported classification model '{model_name}'. Options: {list(ARCHITECTURES)}")
    return ARCHITECTURES[model_name]
