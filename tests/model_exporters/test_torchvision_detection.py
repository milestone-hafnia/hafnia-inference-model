"""torchvision detection models -> ONNX bundle, checked against the original torch models.

The bundles run on the ONNX Runtime flavors, the reference engine. Parity is checked by matching detections on
IoU and class (not by index), on the micro COCO images.
"""

from typing import List

import cv2
import numpy as np
import pytest
from hafnia.dataset.primitives import Bbox
from helpers import assert_detections_match

from inference_model_core import inference_model_from_path, read_model_info

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
from model_exporters.torchvision_detection import IGNORED_CATEGORIES  # noqa: E402

pytestmark = pytest.mark.slow


def torch_detections(model, categories: List[str], image: np.ndarray) -> List[Bbox]:
    """Run the original torch model and convert its detections to normalized 'Bbox' primitives."""
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        output = model([image_tensor])[0]
    height, width = image.shape[:2]
    detections = []
    for (x0, y0, x1, y1), label, score in zip(
        output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist(), strict=True
    ):
        if categories[label] in IGNORED_CATEGORIES:
            continue
        detections.append(
            Bbox(
                top_left_x=x0 / width,
                top_left_y=y0 / height,
                width=(x1 - x0) / width,
                height=(y1 - y0) / height,
                class_name=categories[label],
                confidence=score,
            )
        )
    return detections


def test_model_info(ssd, frcnn, micro_coco):
    ssd_info, frcnn_info = read_model_info(ssd[2]), read_model_info(frcnn[2])
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()

    for info in (ssd_info, frcnn_info):
        assert info.framework == "torchvision"
        assert info.tasks[0].get_class_names() == coco_class_names
        assert len(info.runtime.postprocessing.params["label_to_class_idx"]) == 91

    assert ssd_info.runtime.input.tensors[0].shape == [None, 3, 320, 320]
    assert ssd_info.runtime.preprocessing.resize is not None
    assert frcnn_info.runtime.input.tensors[0].shape == [3, None, None]
    assert frcnn_info.runtime.preprocessing.in_graph == ["resize", "normalize"]


def test_ssd_matches_torch_without_resize(ssd, images):
    """With an image already at the model input size, resizing is an identity on both sides, so the only
    differences are numerical (ONNX Runtime vs torch, NumPy vs torch postprocessing)."""
    torch_model, categories, path_bundle = ssd
    model = inference_model_from_path(path_bundle)
    for image in images:
        image_320 = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
        n_checked = assert_detections_match(
            reference=torch_detections(torch_model, categories, image_320),
            predicted=model.predict(image_320),
            min_score=0.05,
            min_iou=0.99,
            max_score_diff=1e-3,
        )
        assert n_checked > 0


def test_ssd_matches_torch_on_original_images(ssd, images):
    """Resizing outside the graph (OpenCV on uint8) differs slightly from torchvision's in-graph resize."""
    torch_model, categories, path_bundle = ssd
    model = inference_model_from_path(path_bundle)
    n_checked = sum(
        assert_detections_match(
            reference=torch_detections(torch_model, categories, image),
            predicted=model.predict(image),
            min_score=0.3,
            min_iou=0.97,
            max_score_diff=0.03,
        )
        for image in images
    )
    assert n_checked >= len(images)


def test_frcnn_matches_torch(frcnn, images):
    """Same detections as the original model in both directions (nothing missing, nothing extra). Same input
    size on both sides, so differences are numerical only."""
    torch_model, categories, path_bundle = frcnn
    model = inference_model_from_path(path_bundle)
    settings = {"min_score": model.params.score_threshold, "min_iou": 0.99, "max_score_diff": 1e-3}
    for image in images:
        reference, predicted = torch_detections(torch_model, categories, image), model.predict(image)
        assert assert_detections_match(reference, predicted, **settings) > 0
        assert_detections_match(predicted, reference, **settings)


@pytest.mark.parametrize("height, width", [(400, 640), (480, 640), (500, 375), (320, 320), (300, 401)])
def test_frcnn_matches_torch_across_input_sizes(frcnn, images, height, width):
    """The in-graph resize must give the same size as torch. 400 x 640 scales by exactly 0.8, which the
    float32 size computation of the traced graph got wrong (see '_resize_image_export' in the exporter)."""
    torch_model, categories, path_bundle = frcnn
    model = inference_model_from_path(path_bundle)
    image = cv2.resize(images[0], (width, height), interpolation=cv2.INTER_LINEAR)
    reference, predicted = torch_detections(torch_model, categories, image), model.predict(image)
    settings = {"min_score": model.params.score_threshold, "min_iou": 0.99, "max_score_diff": 1e-3}
    assert_detections_match(reference, predicted, **settings)
    assert_detections_match(predicted, reference, **settings)


@pytest.mark.parametrize("fill_value", [0, 128, 255, None])
def test_frcnn_images_without_proposals(frcnn, fill_value):
    """Blank and noise images give no RPN proposals in the original model. The exported graph must run the
    zero-boxes path (see '_export_safe_ops' in the exporter) and return nothing, like the original."""
    torch_model, categories, path_bundle = frcnn
    if fill_value is None:
        image = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    else:
        image = np.full((300, 400, 3), fill_value, dtype=np.uint8)
    model = inference_model_from_path(path_bundle)
    raw_outputs = model.predict_raw(model.preprocess(image)[0])

    assert torch_detections(torch_model, categories, image) == []
    assert {name: output.shape for name, output in raw_outputs.items()} == {
        "boxes": (0, 4),
        "scores": (0,),
        "labels": (0,),
    }
    assert model.predict(image) == []
