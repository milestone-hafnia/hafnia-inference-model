"""Choose the exporter for a pretrained model name or a trainer archive.

Exporter modules are imported only when used, so e.g. exporting a torchvision model doesn't need 'rfdetr'.
"""

import importlib
from pathlib import Path
from typing import Dict, List, Union

from model_exporters.onnx_export import DEFAULT_OPSET_VERSION
from model_exporters.trainer_archive import read_trainer_config

# Model name -> exporter module. Each module has 'export_pretrained_*' and, for trainer models,
# 'export_trainer_archive_*' functions (see '_EXPORT_FUNCTIONS').
_PRETRAINED_MODELS: Dict[str, str] = {
    "ssdlite320_mobilenet_v3_large": "model_exporters.torchvision_detection",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "model_exporters.torchvision_detection",
    "resnet18": "model_exporters.torchvision_classification",
    "RFDETRNano": "model_exporters.rfdetr_detection",
    "RFDETRSmall": "model_exporters.rfdetr_detection",
    "RFDETRMedium": "model_exporters.rfdetr_detection",
    "RFDETRLarge": "model_exporters.rfdetr_detection",
}
_EXPORT_FUNCTIONS = {
    "model_exporters.torchvision_detection": ("export_pretrained_torchvision_detection", None),
    "model_exporters.torchvision_classification": (
        "export_pretrained_torchvision_classification",
        "export_trainer_archive_classification",
    ),
    "model_exporters.rfdetr_detection": ("export_pretrained_rfdetr_detection", "export_trainer_archive_rfdetr"),
}


def pretrained_model_names() -> List[str]:
    return list(_PRETRAINED_MODELS)


def export_pretrained(
    model_name: str,
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    """Export a pretrained model by name (see 'pretrained_model_names') to an ONNX bundle."""
    if model_name not in _PRETRAINED_MODELS:
        raise ValueError(f"Unknown pretrained model '{model_name}'. Options: {pretrained_model_names()}")
    module_name = _PRETRAINED_MODELS[model_name]
    export = getattr(importlib.import_module(module_name), _EXPORT_FUNCTIONS[module_name][0])
    return export(model_name, path_bundle, opset_version=opset_version, overwrite=overwrite)


def export_trainer_archive(
    path_archive: Union[str, Path],
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    """Export a model archive of trainer-classification or trainer-object-detection to an ONNX bundle."""
    model_name = read_trainer_config(path_archive)["name"]
    trainer_models = {name: module for name, module in _PRETRAINED_MODELS.items() if _EXPORT_FUNCTIONS[module][1]}
    if model_name not in trainer_models:
        raise ValueError(f"Can't export trainer model '{model_name}'. Supported: {list(trainer_models)}")
    module_name = trainer_models[model_name]
    export = getattr(importlib.import_module(module_name), str(_EXPORT_FUNCTIONS[module_name][1]))
    return export(path_archive, path_bundle, opset_version=opset_version, overwrite=overwrite)
