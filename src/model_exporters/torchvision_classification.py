"""Model bundles of torchvision classification models (ResNet18).

Sources: a model archive of trainer-classification (e.g. a platform checkpoint) or pretrained ImageNet weights by
name. The bundle is a graph of components:

    load_image -> resize -> to_tensor -> network -> to_classification

The network is a torch component ('torchvision_classifier'): input '[N, 3, S, S]', output the class logits. It
converts to ONNX with a dynamic batch axis, and from there to OpenVINO (see 'inference_model_core.convert').

Preprocessing follows trainer-classification's training transform: resize to 'image_size' x 'image_size'
(torchvision 'Resize', antialiased on tensors), scale to [0, 1], ImageNet mean/std.
"""

import copy
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torchvision
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torchvision.models import ResNet18_Weights, WeightsEnum, resnet18

from inference_model_core import (
    ConversionInfo,
    GraphInfo,
    ModelBundleInfo,
    NetworkParams,
    TensorInfo,
    register_component,
    write_bundle,
)
from inference_model_core.components import ClassificationFromLogits, ImageToTensor, LoadImage, ResizeImage
from inference_model_core.convert import TARGET_ONNX, TARGET_OPENVINO, TARGET_OPENVINO_INT8
from inference_model_torch.component import OnnxExport, TorchNetworkComponent, load_state_dict_file
from inference_model_torch.onnx_export import DEFAULT_OPSET_VERSION, batch_axes, onnx_export, onnx_options
from model_exporters.trainer_archive import open_trainer_archive

FRAMEWORK = "torchvision"
WEIGHTS_FILE = "network/weights.pth"
TASK_NAME = "classification"
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Architecture name (as in trainer-classification's 'model_config.json') -> (builder, ImageNet weights)
ARCHITECTURES: Dict[str, Tuple[Callable[..., nn.Module], WeightsEnum]] = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
}


class TorchvisionClassifierParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    architecture: str = Field(description=f"torchvision architecture. Options: {list(ARCHITECTURES)}")
    num_classes: int = Field(gt=0)
    image_size: int = Field(gt=0, description="Square input size the model was trained on")


@register_component
class TorchvisionClassifier(TorchNetworkComponent):
    """A torchvision classification network: 'images' (N, 3, S, S) -> 'logits' (N, classes)."""

    type = "torchvision_classifier"
    params_model = TorchvisionClassifierParams
    params: TorchvisionClassifierParams
    input_ports = ("images",)
    output_ports = ("logits",)

    @classmethod
    def build_model(cls, params: TorchvisionClassifierParams, path_weights: Path, device: torch.device) -> nn.Module:  # type: ignore[override]
        builder, _ = _architecture(params.architecture)
        model = builder(weights=None, num_classes=params.num_classes)
        model.load_state_dict(load_state_dict_file(path_weights), strict=True)
        return model.to(device).eval()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"logits": self.model(inputs["images"])}

    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        options = onnx_options(options, defaults={"opset_version": DEFAULT_OPSET_VERSION, "dynamic_batch": True})
        size, dynamic_batch = self.params.image_size, options["dynamic_batch"]
        onnx_export(
            copy.deepcopy(self.model).cpu(),
            (torch.zeros(2 if dynamic_batch else 1, 3, size, size),),
            path_onnx,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes=batch_axes(["images", "logits"], dynamic_batch),
            opset_version=options["opset_version"],
        )
        batch = None if dynamic_batch else 1
        params = NetworkParams(
            input_tensors=[TensorInfo(name="images", shape=[batch, 3, size, size])],
            output_tensors=[TensorInfo(name="logits", shape=[batch, self.params.num_classes])],
        )
        return OnnxExport(params=params, batched=True)


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
        model.load_state_dict(load_state_dict_file(path_weights))
    return model.eval(), task, int(config.get("image_size", DEFAULT_IMAGE_SIZE)), model_name


def create_pretrained_torchvision_classification_bundle(
    model_name: str, path_bundle: Union[str, Path], overwrite: bool = False
) -> Path:
    model, task, image_size = load_pretrained_model(model_name)
    return create_torchvision_classification_bundle(
        model,
        architecture=model_name,
        task=task,
        image_size=image_size,
        path_bundle=path_bundle,
        name=model_name,
        variant=model_name,
        overwrite=overwrite,
        provenance={"weights": str(ARCHITECTURES[model_name][1])},
    )


def create_trainer_archive_classification_bundle(
    path_archive: Union[str, Path], path_bundle: Union[str, Path], overwrite: bool = False
) -> Path:
    model, task, image_size, model_name = load_trainer_archive(path_archive)
    return create_torchvision_classification_bundle(
        model,
        architecture=model_name,
        task=task,
        image_size=image_size,
        path_bundle=path_bundle,
        name=model_name,
        variant=model_name,
        overwrite=overwrite,
        provenance={"trainer_archive": Path(path_archive).name},
    )


def create_torchvision_classification_bundle(
    model: nn.Module,
    architecture: str,
    task: TaskInfo,
    image_size: int,
    path_bundle: Union[str, Path],
    name: str,
    variant: Optional[str] = None,
    overwrite: bool = False,
    provenance: Optional[dict] = None,
) -> Path:
    """Write a torchvision classification model as a model bundle with a torch network.

    Args:
        model: Classification model with one logit per class of 'task'. It is not modified.
        architecture: torchvision architecture of 'model' (see 'ARCHITECTURES'). The bundle stores the weights
            and rebuilds the module from this name.
        task: Classification task with the class names in logit order.
        image_size: Square input size the model was trained on.
        path_bundle: Output bundle, a '.zip' path or a directory.
        name: Model name stored in the model info.
        variant: Informational variant stored in the model info.
        overwrite: Replace an existing bundle.
        provenance: Extra provenance information stored with the network component.
    """
    _architecture(architecture)
    resize = {
        "mode": "stretch",
        "height": image_size,
        "width": image_size,
        "interpolation": "linear",
        "antialias": True,
    }
    components = [
        LoadImage.create_info("load_image", inputs={"image": "image"}, outputs={"image": "image_rgb"}),
        ResizeImage.create_info(
            "resize",
            inputs={"image": "image_rgb"},
            outputs={"image": "image_resized", "transform": "image_transform"},
            params=resize,
        ),
        ImageToTensor.create_info(
            "to_tensor",
            inputs={"image": "image_resized"},
            outputs={"tensor": "input_tensor"},
            params={"scale": 1 / 255, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        ),
        TorchvisionClassifier.create_info(
            "network",
            inputs={"images": "input_tensor"},
            outputs={"logits": "logits"},
            params={
                "architecture": architecture,
                "num_classes": len(task.get_class_names() or []),
                "image_size": image_size,
            },
            files=[WEIGHTS_FILE],
            batched=True,
            conversions={
                TARGET_ONNX: ConversionInfo(options={"opset_version": DEFAULT_OPSET_VERSION, "dynamic_batch": True}),
                TARGET_OPENVINO: ConversionInfo(),
                TARGET_OPENVINO_INT8: ConversionInfo(),
            },
            provenance={
                "created_by": __name__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                **(provenance or {}),
            },
        ),
        ClassificationFromLogits.create_info(
            "to_classification",
            inputs={"logits": "logits"},
            outputs={"predictions": "predictions"},
            params={"task_name": task.name, "score_threshold": 0.0},
        ),
    ]
    model_info = ModelBundleInfo(
        name=name,
        framework=FRAMEWORK,
        variant=variant,
        tasks=[task],
        graph=GraphInfo(inputs=["image"], outputs=["predictions"], components=components),
    )
    with tempfile.TemporaryDirectory(prefix="classification_bundle_") as tmp_dir:
        path_weights = Path(tmp_dir) / "weights.pth"
        torch.save(model.state_dict(), path_weights)
        return write_bundle(path_bundle, model_info, files={WEIGHTS_FILE: path_weights}, overwrite=overwrite)


def _architecture(model_name: str) -> Tuple[Callable[..., nn.Module], WeightsEnum]:
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unsupported classification model '{model_name}'. Options: {list(ARCHITECTURES)}")
    return ARCHITECTURES[model_name]
