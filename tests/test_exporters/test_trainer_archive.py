import json
import zipfile

import pytest

from hafnia_model_bundle.core import ModelBundle
from hafnia_model_bundle.exporters.trainer_archive import create_bundle_from_trainer_archive, is_trainer_archive


def write_archive(path, config: dict):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model_config.json", json.dumps(config))
    return path


def test_every_pretrained_model_is_registered():
    """The entry points in 'pyproject.toml' list every pretrained model of the family modules."""
    pytest.importorskip("torch")
    pytest.importorskip("rfdetr")
    from hafnia_model_bundle.exporters import rfdetr_detection, torchvision_classification, torchvision_detection

    models = {
        *torchvision_detection.PRETRAINED_MODELS,
        *torchvision_classification.ARCHITECTURES,
        *rfdetr_detection.MODEL_CLASSES,
    }
    assert models <= set(ModelBundle.list_pretrained())
    with pytest.raises(ValueError, match="Unknown pretrained model"):
        ModelBundle.from_pretrained("yolo", "bundle.zip")


def test_unsupported_trainer_archive(tmp_path):
    path_archive = write_archive(tmp_path / "seg.zip", {"name": "RFDETRSegNano", "model_weight_path": "w.pth"})
    assert is_trainer_archive(path_archive)
    assert not is_trainer_archive(tmp_path)
    with pytest.raises(ValueError, match="trainer model 'RFDETRSegNano': it is not the name of a pretrained model"):
        create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle.zip")


def test_trainer_archive_of_a_family_without_archives(tmp_path):
    """The family is found through the pretrained entry point of the model name. It reads trainer archives when
    its module has a 'create_trainer_archive_bundle' function."""
    pytest.importorskip("torch")
    config = {"name": "ssdlite320_mobilenet_v3_large", "model_weight_path": "w.pth"}
    path_archive = write_archive(tmp_path / "ssd.zip", config)
    with pytest.raises(
        ValueError, match=r"'hafnia_model_bundle.exporters\.torchvision_detection' has no 'create_trainer_archive"
    ):
        create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle.zip")


def test_trainer_archive_without_weights(tmp_path):
    pytest.importorskip("torch")
    path_archive = write_archive(tmp_path / "empty.zip", {"name": "resnet18", "model_weight_path": None})
    with pytest.raises(ValueError, match="has no weights"):
        create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle.zip")
