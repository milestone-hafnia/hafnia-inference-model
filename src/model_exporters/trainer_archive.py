"""Model archives of the Hafnia trainer packages (trainer-classification, trainer-object-detection).

An archive is a zip with 'model_config.json' (model name, task with class names and model specific settings)
and the weights file it names in 'model_weight_path'. Platform checkpoints of those trainers use this format.

The model name of an archive is the name of a pretrained model ('ModelBundle.list_pretrained'). Its entry point
names the module of the model family, and a family that reads trainer archives has a function
'create_trainer_archive_bundle(path_archive, path_bundle, overwrite) -> ModelBundle' in that module. So a family is
imported only when one of its archives is read, and no table of trainer models has to be kept.
"""

import contextlib
import importlib
import json
import tempfile
import zipfile
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union

from inference_model_core.model_bundle import PRETRAINED_ENTRY_POINT_GROUP, ModelBundle

MODEL_CONFIG_FILENAME = "model_config.json"
TRAINER_ARCHIVE_FUNCTION = "create_trainer_archive_bundle"


def create_bundle_from_trainer_archive(
    path_archive: Union[str, Path], path_bundle: Union[str, Path], overwrite: bool = False
) -> ModelBundle:
    """Write a model archive of trainer-classification or trainer-object-detection as a model bundle with a
    torch network. Returns the written bundle."""
    model_name = read_trainer_config(path_archive)["name"]
    modules = sorted(
        {entry_point.module for entry_point in entry_points(group=PRETRAINED_ENTRY_POINT_GROUP, name=model_name)}
    )
    if not modules:
        raise ValueError(
            f"Can't create a bundle of trainer model '{model_name}': it is not the name of a pretrained model. "
            f"Pretrained models: {ModelBundle.list_pretrained()}"
        )
    create = getattr(importlib.import_module(modules[0]), TRAINER_ARCHIVE_FUNCTION, None)
    if create is None:
        raise ValueError(
            f"Can't create a bundle of trainer model '{model_name}': its model family '{modules[0]}' has no "
            f"'{TRAINER_ARCHIVE_FUNCTION}' function."
        )
    return create(path_archive, path_bundle, overwrite=overwrite)


def is_trainer_archive(path: Union[str, Path]) -> bool:
    path = Path(path)
    if not (path.is_file() and zipfile.is_zipfile(path)):
        return False
    with zipfile.ZipFile(path) as archive:
        return MODEL_CONFIG_FILENAME in archive.namelist()


def read_trainer_config(path_archive: Union[str, Path]) -> Dict[str, Any]:
    with zipfile.ZipFile(path_archive) as archive:
        return json.loads(archive.read(MODEL_CONFIG_FILENAME))


@contextlib.contextmanager
def open_trainer_archive(path_archive: Union[str, Path]) -> Iterator[Tuple[Dict[str, Any], Path]]:
    """Yield the model config and the path of the extracted weights file, which exists inside the 'with'."""
    config = read_trainer_config(path_archive)
    weight_name = config.get("model_weight_path")
    if not weight_name:
        raise ValueError(f"Model archive '{path_archive}' has no weights ('model_weight_path' is empty).")
    with tempfile.TemporaryDirectory(prefix="trainer_archive_") as tmp_dir, zipfile.ZipFile(path_archive) as archive:
        archive.extract(Path(weight_name).name, path=tmp_dir)
        yield config, Path(tmp_dir) / Path(weight_name).name
