import json
import zipfile

import pytest

from model_exporters.registry import (
    create_bundle_from_trainer_archive,
    create_pretrained_bundle,
    pretrained_model_names,
)
from model_exporters.trainer_archive import is_trainer_archive


def write_archive(path, config: dict):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model_config.json", json.dumps(config))
    return path


def test_pretrained_model_names():
    assert {"ssdlite320_mobilenet_v3_large", "resnet18", "RFDETRNano"} <= set(pretrained_model_names())
    with pytest.raises(ValueError, match="Unknown pretrained model"):
        create_pretrained_bundle("yolo", "bundle.zip")


def test_unsupported_trainer_archive(tmp_path):
    path_archive = write_archive(tmp_path / "seg.zip", {"name": "RFDETRSegNano", "model_weight_path": "w.pth"})
    assert is_trainer_archive(path_archive)
    assert not is_trainer_archive(tmp_path)
    with pytest.raises(ValueError, match="Can't create a bundle of trainer model 'RFDETRSegNano'"):
        create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle.zip")


def test_trainer_archive_without_weights(tmp_path):
    pytest.importorskip("torch")
    path_archive = write_archive(tmp_path / "empty.zip", {"name": "resnet18", "model_weight_path": None})
    with pytest.raises(ValueError, match="has no weights"):
        create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle.zip")
