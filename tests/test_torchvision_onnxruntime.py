"""End-to-end: pretrained torchvision model -> ONNX bundle -> ONNX Runtime flavor, checked against torch.

Parity is checked by matching detections on IoU and class (not by index), on the micro COCO images.
"""

import math
from typing import List

import cv2
import numpy as np
import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark
from hafnia.dataset.primitives import Bbox
from helpers import assert_detections_match

from inference_model_core import inference_model_from_path, read_model_info

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
from inference_model_onnxruntime.flavors import (  # noqa: E402
    OnnxRuntimeTorchvisionDetection,
    OnnxRuntimeTorchvisionSSD,
)
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


def test_bundles_select_flavors_via_entry_points(ssd, frcnn):
    assert type(inference_model_from_path(ssd[2])) is OnnxRuntimeTorchvisionSSD
    assert type(inference_model_from_path(frcnn[2])) is OnnxRuntimeTorchvisionDetection


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
    assert frcnn_info.runtime.provenance["rpn_score_threshold"] == 0.0


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


def test_frcnn_reproduces_all_torch_detections(frcnn, images):
    """The exported graph keeps every detection of the original model (score >= 0.05), see the RPN score
    threshold in the exporter. Same input size on both sides, so differences are numerical only."""
    torch_model, categories, path_bundle = frcnn
    model = inference_model_from_path(path_bundle)
    for image in images:
        n_checked = assert_detections_match(
            reference=torch_detections(torch_model, categories, image),
            predicted=model.predict(image),
            min_score=model.params.score_threshold,
            min_iou=0.99,
            max_score_diff=1e-3,
        )
        assert n_checked > 0


@pytest.mark.parametrize("fill_value", [0, 128, 255])
def test_frcnn_handles_images_without_proposals(frcnn, fill_value):
    """Blank images give no RPN proposals in the original model, which must not crash the exported graph."""
    model = inference_model_from_path(frcnn[2])
    predictions = model.predict(np.full((300, 400, 3), fill_value, dtype=np.uint8))
    assert all(p.confidence >= model.params.score_threshold for p in predictions)


def test_ssd_batching(ssd, images):
    model = inference_model_from_path(ssd[2])
    assert model.supports_batching()

    batch_predictions = model.predict_batch(images)
    for image, predictions in zip(images, batch_predictions, strict=True):
        assert_detections_match(model.predict(image), predictions, min_score=0.01, min_iou=0.999, max_score_diff=1e-4)


def test_frcnn_runs_image_by_image(frcnn, images):
    model = inference_model_from_path(frcnn[2])
    assert not model.supports_batching()
    assert len(model.predict_batch(images)) == len(images)


def test_score_threshold_is_a_runtime_setting(ssd, images):
    default_model = inference_model_from_path(ssd[2])
    strict_model = inference_model_from_path(ssd[2], postprocessing_params={"score_threshold": 0.5})

    default_predictions, strict_predictions = default_model.predict(images[0]), strict_model.predict(images[0])
    assert len(strict_predictions) < len(default_predictions)
    assert strict_predictions and all(p.confidence > 0.5 for p in strict_predictions)
    assert strict_predictions == [p for p in default_predictions if p.confidence > 0.5]


def test_write_model_round_trip(ssd, images, tmp_path):
    model = inference_model_from_path(ssd[2])
    path_copy = model.write_model(tmp_path / "copy")
    assert inference_model_from_path(path_copy).predict(images[0]) == model.predict(images[0])


def test_engine_options(ssd):
    with pytest.raises(ValueError, match="Unsupported device"):
        inference_model_from_path(ssd[2], device="TPU")
    with pytest.raises(TypeError, match="Unknown ONNX Runtime engine options"):
        inference_model_from_path(ssd[2], unknown_option=1)


def test_hafnia_benchmark_on_micro_coco(ssd, micro_coco):
    model = inference_model_from_path(ssd[2])
    metrics, dataset_predictions = run_benchmark(dataset=micro_coco, model=model, batch_size=2)

    assert "object_detection/predictions" in [task.name for task in dataset_predictions.info.tasks]
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())
    assert max(metrics.values()) > 0
