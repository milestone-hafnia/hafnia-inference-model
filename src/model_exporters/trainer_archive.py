"""Model archives of the Hafnia trainer packages (trainer-classification, trainer-object-detection).

An archive is a zip with 'model_config.json' (model name, task with class names and model specific settings)
and the weights file it names in 'model_weight_path'. Platform checkpoints of those trainers use this format.
"""

import contextlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union

MODEL_CONFIG_FILENAME = "model_config.json"


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
