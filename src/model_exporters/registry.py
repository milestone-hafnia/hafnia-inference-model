"""Choose the bundle helper for a trainer archive.

The helpers write a model bundle with a torch network, the source format of a model. 'ModelBundle.convert' turns it
into an ONNX or OpenVINO bundle. Family modules are imported only when used, so e.g. a torchvision model doesn't
need 'rfdetr'.

Pretrained models are not listed here: they are entry points of 'hafnia.inference_model_pretrained_models' (see
'pyproject.toml') and written with 'ModelBundle.from_pretrained'.
"""

import importlib
from pathlib import Path
from typing import Dict, Union

from inference_model_core.model_bundle import ModelBundle
from model_exporters.trainer_archive import read_trainer_config

# Model name in the trainer archive -> '<family module>:<function>' that writes the archive as a bundle
_TRAINER_MODELS: Dict[str, str] = {
    "resnet18": "model_exporters.torchvision_classification:create_trainer_archive_classification_bundle",
    "RFDETRNano": "model_exporters.rfdetr_detection:create_trainer_archive_rfdetr_bundle",
    "RFDETRSmall": "model_exporters.rfdetr_detection:create_trainer_archive_rfdetr_bundle",
    "RFDETRMedium": "model_exporters.rfdetr_detection:create_trainer_archive_rfdetr_bundle",
    "RFDETRLarge": "model_exporters.rfdetr_detection:create_trainer_archive_rfdetr_bundle",
}


def create_bundle_from_trainer_archive(
    path_archive: Union[str, Path], path_bundle: Union[str, Path], overwrite: bool = False
) -> ModelBundle:
    """Write a model archive of trainer-classification or trainer-object-detection as a model bundle with a
    torch network. Returns the written bundle."""
    model_name = read_trainer_config(path_archive)["name"]
    if model_name not in _TRAINER_MODELS:
        raise ValueError(f"Can't create a bundle of trainer model '{model_name}'. Supported: {list(_TRAINER_MODELS)}")
    module_name, function_name = _TRAINER_MODELS[model_name].split(":")
    create = getattr(importlib.import_module(module_name), function_name)
    return create(path_archive, path_bundle, overwrite=overwrite)
