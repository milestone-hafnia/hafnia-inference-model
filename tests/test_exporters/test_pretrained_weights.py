"""Pretrained weights are downloaded into 'models/<framework>' of the working directory."""

import os

import torch
from rfdetr.assets.model_weights import get_model_cache_dir

from hafnia_inference_model.exporters.pretrained_weights import rfdetr_weights_dir, torchvision_weights_dir


def test_torchvision_weights_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = torch.hub.get_dir()
    with torchvision_weights_dir() as path_dir:
        assert path_dir == tmp_path / "models" / "torchvision"
        assert torch.hub.get_dir() == str(path_dir)
    assert torch.hub.get_dir() == previous


def test_rfdetr_weights_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RF_HOME", raising=False)
    with rfdetr_weights_dir() as path_dir:
        assert path_dir == tmp_path / "models" / "rfdetr"
        assert get_model_cache_dir() == str(path_dir)
    assert "RF_HOME" not in os.environ

    monkeypatch.setenv("RF_HOME", "elsewhere")
    with rfdetr_weights_dir():
        pass
    assert os.environ["RF_HOME"] == "elsewhere"
