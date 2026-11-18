import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))  # Make 'helpers' importable from test modules

from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from helpers import PATH_MICRO_COCO, DummyDetector

from inference_model_core import register_flavor
from inference_model_core.registry import unregister_flavor

SSD_NAME = "ssdlite320_mobilenet_v3_large"
FRCNN_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def dummy_flavor():
    register_flavor(DummyDetector)
    yield DummyDetector
    unregister_flavor(DummyDetector.flavor_key())


@pytest.fixture(scope="session")
def micro_coco() -> HafniaDataset:
    return HafniaDataset.from_path(PATH_MICRO_COCO)


@pytest.fixture(scope="session")
def images(micro_coco) -> List[np.ndarray]:
    return [Sample(**sample).read_image() for sample in micro_coco]


def _export_onnx(tmp_path_factory, model_name):
    """Export a pretrained torchvision model to an ONNX bundle. Returns (torch model, categories, bundle path)."""
    pytest.importorskip("torch")
    from model_exporters.torchvision_detection import export_torchvision_detection, load_pretrained_model

    model, categories = load_pretrained_model(model_name)
    path_bundle = tmp_path_factory.mktemp(model_name) / f"{model_name}.zip"
    export_torchvision_detection(model, categories, path_bundle, name=model_name, variant=model_name)
    return model, categories, path_bundle


# Session scoped: the exports are shared by the end-to-end tests of all engines
@pytest.fixture(scope="session")
def ssd(tmp_path_factory):
    return _export_onnx(tmp_path_factory, SSD_NAME)


@pytest.fixture(scope="session")
def frcnn(tmp_path_factory):
    return _export_onnx(tmp_path_factory, FRCNN_NAME)
