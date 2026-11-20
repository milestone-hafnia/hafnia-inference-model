"""Choose the bundle helper for a pretrained model name or a trainer archive.

The helpers write a model bundle with a torch network, the source format of a model. 'convert_bundle' turns it
into an ONNX or OpenVINO bundle. Family modules are imported only when used, so e.g. a torchvision model doesn't
need 'rfdetr'.
"""

import importlib
from pathlib import Path
from typing import Dict, List, Union

from model_exporters.trainer_archive import read_trainer_config

# Model name -> family module. Each module has 'create_pretrained_*_bundle' and, for trainer models,
# 'create_trainer_archive_*_bundle' functions (see '_BUNDLE_FUNCTIONS').
_PRETRAINED_MODELS: Dict[str, str] = {
    "ssdlite320_mobilenet_v3_large": "model_exporters.torchvision_detection",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "model_exporters.torchvision_detection",
    "resnet18": "model_exporters.torchvision_classification",
    "RFDETRNano": "model_exporters.rfdetr_detection",
    "RFDETRSmall": "model_exporters.rfdetr_detection",
    "RFDETRMedium": "model_exporters.rfdetr_detection",
    "RFDETRLarge": "model_exporters.rfdetr_detection",
}
_BUNDLE_FUNCTIONS = {
    "model_exporters.torchvision_detection": ("create_pretrained_torchvision_detection_bundle", None),
    "model_exporters.torchvision_classification": (
        "create_pretrained_torchvision_classification_bundle",
        "create_trainer_archive_classification_bundle",
    ),
    "model_exporters.rfdetr_detection": (
        "create_pretrained_rfdetr_detection_bundle",
        "create_trainer_archive_rfdetr_bundle",
    ),
}


def pretrained_model_names() -> List[str]:
    return list(_PRETRAINED_MODELS)


def create_pretrained_bundle(model_name: str, path_bundle: Union[str, Path], overwrite: bool = False) -> Path:
    """Write a pretrained model by name (see 'pretrained_model_names') as a model bundle with a torch network."""
    if model_name not in _PRETRAINED_MODELS:
        raise ValueError(f"Unknown pretrained model '{model_name}'. Options: {pretrained_model_names()}")
    module_name = _PRETRAINED_MODELS[model_name]
    create = getattr(importlib.import_module(module_name), _BUNDLE_FUNCTIONS[module_name][0])
    return create(model_name, path_bundle, overwrite=overwrite)


def create_bundle_from_trainer_archive(
    path_archive: Union[str, Path], path_bundle: Union[str, Path], overwrite: bool = False
) -> Path:
    """Write a model archive of trainer-classification or trainer-object-detection as a model bundle with a
    torch network."""
    model_name = read_trainer_config(path_archive)["name"]
    trainer_models = {name: module for name, module in _PRETRAINED_MODELS.items() if _BUNDLE_FUNCTIONS[module][1]}
    if model_name not in trainer_models:
        raise ValueError(f"Can't create a bundle of trainer model '{model_name}'. Supported: {list(trainer_models)}")
    module_name = trainer_models[model_name]
    create = getattr(importlib.import_module(module_name), str(_BUNDLE_FUNCTIONS[module_name][1]))
    return create(path_archive, path_bundle, overwrite=overwrite)
