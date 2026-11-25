"""'ModelBundle.from_pretrained': pretrained models are entry points that are imported only when used."""

import json
import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from helpers import DUMMY_DETECTIONS, DUMMY_NETWORK_FILE, make_bundle

from hafnia_inference_model.core import ModelBundle
from hafnia_inference_model.core.model_bundle import PRETRAINED_ENTRY_POINT_GROUP

CREATE_DUMMY = f"{__name__}:create_dummy_bundle"


def create_dummy_bundle(name: str, path_bundle: Path, overwrite: bool = False) -> ModelBundle:
    """A function as it is registered for a pretrained model."""
    bundle = make_bundle().model_copy(update={"name": name})
    files = {DUMMY_NETWORK_FILE: json.dumps(DUMMY_DETECTIONS).encode()}
    return bundle.write(path_bundle, files=files, overwrite=overwrite)


def create_path_only(name: str, path_bundle: Path, overwrite: bool = False) -> Optional[Path]:
    return create_dummy_bundle(name, path_bundle, overwrite=overwrite).path


@pytest.fixture
def register_pretrained(monkeypatch):
    """Replace the registered pretrained models by: model name -> the '<module>:<function>' it is registered with."""

    def register(models: Dict[str, List[str]]) -> None:
        registered = [
            EntryPoint(name, value, PRETRAINED_ENTRY_POINT_GROUP) for name, values in models.items() for value in values
        ]

        def entry_points(group: str, name: Optional[str] = None) -> List[EntryPoint]:
            assert group == PRETRAINED_ENTRY_POINT_GROUP
            return [entry_point for entry_point in registered if name is None or entry_point.name == name]

        monkeypatch.setattr("hafnia_inference_model.core.model_bundle.entry_points", entry_points)

    return register


def test_from_pretrained_writes_the_bundle(tmp_path, register_pretrained):
    register_pretrained({"dummy-b": [CREATE_DUMMY], "dummy-a": [CREATE_DUMMY]})
    assert ModelBundle.list_pretrained() == ["dummy-a", "dummy-b"]

    bundle = ModelBundle.from_pretrained("dummy-a", tmp_path / "model.zip")
    assert bundle.name == "dummy-a"
    assert bundle.path == tmp_path / "model.zip"
    assert ModelBundle.from_path(bundle.path) == bundle

    with pytest.raises(FileExistsError):
        ModelBundle.from_pretrained("dummy-a", tmp_path / "model.zip")
    assert ModelBundle.from_pretrained("dummy-a", tmp_path / "model.zip", overwrite=True) == bundle


def test_unknown_pretrained_model(tmp_path, register_pretrained):
    register_pretrained({"dummy": [CREATE_DUMMY]})
    with pytest.raises(ValueError, match=r"Unknown pretrained model 'yolo'. Available: \['dummy'\]"):
        ModelBundle.from_pretrained("yolo", tmp_path / "model.zip")


def test_a_model_name_is_registered_once(tmp_path, register_pretrained):
    register_pretrained({"dummy": [CREATE_DUMMY, "other_package:create_dummy_bundle"]})
    assert ModelBundle.list_pretrained() == ["dummy"]
    with pytest.raises(ValueError, match="'dummy' is registered more than once"):
        ModelBundle.from_pretrained("dummy", tmp_path / "model.zip")

    register_pretrained({"dummy": [CREATE_DUMMY, CREATE_DUMMY]})  # The same entry listed twice is one model
    assert ModelBundle.from_pretrained("dummy", tmp_path / "model.zip").name == "dummy"


def test_missing_dependencies_of_a_pretrained_model(tmp_path, register_pretrained):
    register_pretrained({"dummy": ["framework_that_is_not_installed:create_bundle"]})
    assert ModelBundle.list_pretrained() == ["dummy"]  # Listed without importing it
    with pytest.raises(ImportError, match=r"'dummy' .* can not be imported: .* Install its dependencies"):
        ModelBundle.from_pretrained("dummy", tmp_path / "model.zip")


def test_a_pretrained_function_returns_the_bundle(tmp_path, register_pretrained):
    register_pretrained({"dummy": [f"{__name__}:create_path_only"]})
    with pytest.raises(TypeError, match="has to return the written 'ModelBundle'"):
        ModelBundle.from_pretrained("dummy", tmp_path / "model.zip")


def test_pretrained_models_are_listed_without_their_frameworks():
    """The models of this repo are registered in 'pyproject.toml', and listing them imports no framework."""
    code = (
        "import sys; from hafnia_inference_model.core import ModelBundle; names = ModelBundle.list_pretrained(); "
        "assert {'ssdlite320_mobilenet_v3_large', 'resnet18', 'RFDETRNano'} <= set(names), names; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
