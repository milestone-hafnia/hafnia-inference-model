"""Bundles of ResNet18, checked against the original torch model (pretrained and a trainer archive): the bundle
with the torch network and its conversion to ONNX."""

import json
import zipfile

import cv2
import numpy as np
import pytest
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification

from inference_model_core import GraphInferenceModel, ModelBundle, NetworkParams

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
import torchvision.transforms.functional as F  # noqa: E402
from torchvision.models import resnet18 as build_resnet18  # noqa: E402

from model_exporters.registry import create_bundle_from_trainer_archive  # noqa: E402
from model_exporters.torchvision_classification import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

pytestmark = pytest.mark.slow


def torch_probabilities(model, image: np.ndarray, image_size: int) -> np.ndarray:
    """trainer-classification's training transform, then the model and softmax."""
    tensor = F.resize(torch.from_numpy(image).permute(2, 0, 1).float() / 255, [image_size, image_size])
    tensor = F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    with torch.no_grad():
        return model(tensor[None]).softmax(dim=1)[0].numpy()


def test_model_info(resnet18_torch, resnet18):
    torch_info, info = ModelBundle.from_path(resnet18_torch[2]), ModelBundle.from_path(resnet18[2])
    torch_network, network = torch_info.graph.get_component("network"), info.graph.get_component("network")

    assert [c.type for c in torch_info.graph.components] == [
        "load_image",
        "resize_image",
        "image_to_tensor",
        "torchvision_classifier",
        "classification_from_logits",
    ]
    assert (torch_network.type, torch_network.files) == ("torchvision_classifier", ["network/weights.pth"])
    assert torch_network.params == {"architecture": "resnet18", "num_classes": 1000, "image_size": 224}
    assert (network.type, network.batched) == ("onnx_network", True)
    assert NetworkParams.model_validate(network.params).input_tensors[0].shape == [None, 3, 224, 224]

    assert info.framework == "torchvision"
    assert info.tasks[0].primitive is Classification
    assert len(info.tasks[0].get_class_names()) == 1000
    assert info.graph.get_component("resize").params["antialias"]


def test_torch_bundle_matches_torch(resnet18_torch, images):
    """The torch network is rebuilt from the architecture name and the stored weights."""
    torch_model, _, path_bundle = resnet18_torch
    model = GraphInferenceModel.from_path(path_bundle)
    for image in images:
        image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        expected = torch_probabilities(torch_model, image_224, 224)
        (prediction,) = model.predict(image_224)
        assert prediction.class_idx == int(expected.argmax())
        assert prediction.confidence == pytest.approx(float(expected.max()), abs=1e-5)


def test_matches_torch_without_resize(resnet18, images):
    """Images already at the model input size: resizing is an identity on both sides."""
    torch_model, _, path_bundle = resnet18
    model = GraphInferenceModel.from_path(path_bundle)
    for image in images:
        image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        expected = torch_probabilities(torch_model, image_224, 224)
        (prediction,) = model.predict(image_224)
        assert prediction.class_idx == int(expected.argmax())
        assert prediction.confidence == pytest.approx(float(expected.max()), abs=1e-4)


def test_matches_torch_on_original_images(resnet18, images):
    """Our antialiased resize runs on uint8 pixels, torchvision's on floats: probabilities differ slightly."""
    torch_model, _, path_bundle = resnet18
    model = GraphInferenceModel.from_path(path_bundle)
    for image in images:
        expected = torch_probabilities(torch_model, image, 224)
        (prediction,) = model.predict(image)
        assert prediction.class_idx == int(expected.argmax())
        assert prediction.confidence == pytest.approx(float(expected.max()), abs=0.05)


def test_trainer_archive(tmp_path, images):
    """A trainer-classification archive: its classes and image size end up in the bundle."""
    torch.manual_seed(0)
    torch_model = build_resnet18(weights=None, num_classes=3).eval()
    task = TaskInfo.from_class_names(primitive=Classification, class_names=["cat", "dog", "car"], name="classification")
    path_weights = tmp_path / "resnet18.pth"
    torch.save(torch_model.state_dict(), path_weights)
    config = {"name": "resnet18", "task": task.model_dump(mode="json"), "model_weight_path": "resnet18.pth"}
    path_archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(path_archive, "w") as archive:
        archive.writestr("model_config.json", json.dumps({**config, "image_size": 64}))
        archive.write(path_weights, arcname="resnet18.pth")

    path_torch_bundle = create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle_torch.zip")
    torch_info = ModelBundle.from_path(path_torch_bundle)
    assert torch_info.tasks[0].get_class_names() == ["cat", "dog", "car"]
    assert torch_info.graph.get_component("network").params["image_size"] == 64
    assert torch_info.graph.get_component("resize").params["height"] == 64

    path_bundle = ModelBundle.from_path(path_torch_bundle).convert("onnx", tmp_path / "bundle.zip").path
    network = ModelBundle.from_path(path_bundle).graph.get_component("network")
    assert NetworkParams.model_validate(network.params).input_tensors[0].shape == [None, 3, 64, 64]

    for model in (GraphInferenceModel.from_path(path_torch_bundle), GraphInferenceModel.from_path(path_bundle)):
        for image in images:
            image_64 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            expected = torch_probabilities(torch_model, image_64, 64)
            (prediction,) = model.predict(image_64)
            assert prediction.class_name == ["cat", "dog", "car"][int(expected.argmax())]
            assert prediction.confidence == pytest.approx(float(expected.max()), abs=1e-4)
