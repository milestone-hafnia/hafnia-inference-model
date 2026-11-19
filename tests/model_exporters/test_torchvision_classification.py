"""ResNet18 -> ONNX bundle, checked against the original torch model (pretrained and a trainer archive)."""

import json
import zipfile

import cv2
import numpy as np
import pytest
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Classification

from inference_model_core import inference_model_from_path, read_model_info

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
import torchvision.transforms.functional as F  # noqa: E402
from torchvision.models import resnet18 as build_resnet18  # noqa: E402

from model_exporters.registry import export_trainer_archive  # noqa: E402
from model_exporters.torchvision_classification import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

pytestmark = pytest.mark.slow


def torch_probabilities(model, image: np.ndarray, image_size: int) -> np.ndarray:
    """trainer-classification's training transform, then the model and softmax."""
    tensor = F.resize(torch.from_numpy(image).permute(2, 0, 1).float() / 255, [image_size, image_size])
    tensor = F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    with torch.no_grad():
        return model(tensor[None]).softmax(dim=1)[0].numpy()


def test_model_info(resnet18):
    info = read_model_info(resnet18[2])
    assert (info.framework, info.runtime.postprocessing.decoder) == ("torchvision", "classification_logits")
    assert info.tasks[0].primitive is Classification
    assert len(info.tasks[0].get_class_names()) == 1000
    assert info.runtime.input.tensors[0].shape == [None, 3, 224, 224]
    assert info.runtime.preprocessing.resize.antialias


def test_matches_torch_without_resize(resnet18, images):
    """Images already at the model input size: resizing is an identity on both sides."""
    torch_model, _, path_bundle = resnet18
    model = inference_model_from_path(path_bundle)
    for image in images:
        image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        expected = torch_probabilities(torch_model, image_224, 224)
        (prediction,) = model.predict(image_224)
        assert prediction.class_idx == int(expected.argmax())
        assert prediction.confidence == pytest.approx(float(expected.max()), abs=1e-4)


def test_matches_torch_on_original_images(resnet18, images):
    """Our antialiased resize runs on uint8 pixels, torchvision's on floats: probabilities differ slightly."""
    torch_model, _, path_bundle = resnet18
    model = inference_model_from_path(path_bundle)
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

    path_bundle = export_trainer_archive(path_archive, tmp_path / "bundle.zip")
    info = read_model_info(path_bundle)
    assert info.tasks[0].get_class_names() == ["cat", "dog", "car"]
    assert info.runtime.input.tensors[0].shape == [None, 3, 64, 64]

    model = inference_model_from_path(path_bundle)
    for image in images:
        image_64 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        expected = torch_probabilities(torch_model, image_64, 64)
        (prediction,) = model.predict(image_64)
        assert prediction.class_name == ["cat", "dog", "car"][int(expected.argmax())]
        assert prediction.confidence == pytest.approx(float(expected.max()), abs=1e-4)
