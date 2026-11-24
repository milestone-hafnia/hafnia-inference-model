import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))  # Make 'helpers' importable from test modules

from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from helpers import DummyNetwork
from micro_datasets import get_micro_dataset

from inference_model_torch import register_component
from inference_model_torch.component import unregister_component

SSD_NAME = "ssdlite320_mobilenet_v3_large"
FRCNN_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"


@pytest.fixture
def dummy_network():
    """Registers the implementation of the dummy network, so bundles of 'helpers.write_dummy_bundle' can be
    loaded."""
    register_component(DummyNetwork)
    yield DummyNetwork
    unregister_component(DummyNetwork)


@pytest.fixture(scope="session")
def micro_coco() -> HafniaDataset:
    return get_micro_dataset("micro-coco-2017")


@pytest.fixture(scope="session")
def images(micro_coco) -> List[np.ndarray]:
    return [Sample(**sample).read_image() for sample in micro_coco]


# Session scoped: the bundles are shared by the end-to-end tests of all engines. Every model family has a
# '<model>_torch' fixture (the bundle with the torch network, as the bundle helpers write it), a '<model>' fixture
# (converted to ONNX) and a '<model>_ir' fixture (converted to OpenVINO IR). All return
# (original model, categories or task, bundle path).
def _convert(tmp_path_factory, source, target, **kwargs):
    """Convert the bundle of a fixture. Returns the fixture tuple with the converted bundle."""
    from inference_model_core import convert_bundle

    model, labels, path_source = source
    suffix = "_".join([target, *(str(value) for value in kwargs.get("options", {}).values())])
    path_bundle = tmp_path_factory.mktemp(target) / f"{path_source.stem.removesuffix('_torch')}_{suffix}.zip"
    return model, labels, convert_bundle(path_source, path_bundle, target=target, **kwargs)


def _torchvision_detection_bundle(tmp_path_factory, model_name):
    pytest.importorskip("torch")
    from model_exporters.torchvision_detection import create_torchvision_detection_bundle, load_pretrained_model

    model, categories = load_pretrained_model(model_name)
    path_bundle = tmp_path_factory.mktemp(model_name) / f"{model_name}_torch.zip"
    create_torchvision_detection_bundle(
        model, architecture=model_name, categories=categories, path_bundle=path_bundle, name=model_name
    )
    return model, categories, path_bundle


@pytest.fixture(scope="session")
def ssd_torch(tmp_path_factory):
    return _torchvision_detection_bundle(tmp_path_factory, SSD_NAME)


@pytest.fixture(scope="session")
def frcnn_torch(tmp_path_factory):
    return _torchvision_detection_bundle(tmp_path_factory, FRCNN_NAME)


@pytest.fixture(scope="session")
def resnet18_torch(tmp_path_factory):
    """Pretrained ResNet18. Returns (torch model, task, bundle path)."""
    pytest.importorskip("torch")
    from model_exporters.torchvision_classification import (
        create_torchvision_classification_bundle,
        load_pretrained_model,
    )

    model, task, image_size = load_pretrained_model("resnet18")
    path_bundle = tmp_path_factory.mktemp("resnet18") / "resnet18_torch.zip"
    create_torchvision_classification_bundle(
        model, architecture="resnet18", task=task, image_size=image_size, path_bundle=path_bundle, name="resnet18"
    )
    return model, task, path_bundle


@pytest.fixture(scope="session")
def rfdetr_torch(tmp_path_factory):
    """Pretrained RF-DETR Nano. Returns (rfdetr model, categories, bundle path)."""
    pytest.importorskip("rfdetr")
    from model_exporters.rfdetr_detection import create_rfdetr_detection_bundle, load_pretrained_model

    model, categories = load_pretrained_model("RFDETRNano")
    path_bundle = tmp_path_factory.mktemp("rfdetr") / "RFDETRNano_torch.zip"
    create_rfdetr_detection_bundle(
        model,
        model_name="RFDETRNano",
        path_weights=model.model_config.pretrain_weights,
        categories=categories,
        path_bundle=path_bundle,
        name="RFDETRNano",
    )
    return model, categories, path_bundle


@pytest.fixture(scope="session")
def ssd(ssd_torch, tmp_path_factory):
    return _convert(tmp_path_factory, ssd_torch, "onnx")


@pytest.fixture(scope="session")
def frcnn(frcnn_torch, tmp_path_factory):
    return _convert(tmp_path_factory, frcnn_torch, "onnx")


@pytest.fixture(scope="session")
def resnet18(resnet18_torch, tmp_path_factory):
    return _convert(tmp_path_factory, resnet18_torch, "onnx")


@pytest.fixture(scope="session")
def rfdetr(rfdetr_torch, tmp_path_factory):
    return _convert(tmp_path_factory, rfdetr_torch, "onnx")


# OpenVINO IR bundles converted from the ONNX bundles above: only the bundle path
@pytest.fixture(scope="session")
def ssd_ir(ssd, tmp_path_factory):
    pytest.importorskip("openvino")
    return _convert(tmp_path_factory, ssd, "openvino")[2]


@pytest.fixture(scope="session")
def ssd_ir_fp16(ssd, tmp_path_factory):
    pytest.importorskip("openvino")
    return _convert(tmp_path_factory, ssd, "openvino", options={"precision": "fp16"})[2]


@pytest.fixture(scope="session")
def frcnn_ir(frcnn, tmp_path_factory):
    pytest.importorskip("openvino")
    return _convert(tmp_path_factory, frcnn, "openvino")[2]


@pytest.fixture(scope="session")
def resnet18_ir(resnet18, tmp_path_factory):
    pytest.importorskip("openvino")
    return _convert(tmp_path_factory, resnet18, "openvino")[2]


@pytest.fixture(scope="session")
def rfdetr_ir(rfdetr, tmp_path_factory):
    pytest.importorskip("openvino")
    return _convert(tmp_path_factory, rfdetr, "openvino")[2]
