"""ONNX Runtime flavors on bundles exported from torchvision: selection, batching, runtime settings, storage and
Hafnia's benchmark. Parity with torch is tested with the exporter ('tests/model_exporters')."""

import math

import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark
from helpers import assert_detections_match

from inference_model_core import inference_model_from_path

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
from inference_model_onnxruntime.flavors import (
    OnnxRuntimeTorchvisionDetection,
    OnnxRuntimeTorchvisionSSD,
)

pytestmark = pytest.mark.slow


def test_bundles_select_flavors_via_entry_points(ssd, frcnn):
    assert type(inference_model_from_path(ssd[2])) is OnnxRuntimeTorchvisionSSD
    assert type(inference_model_from_path(frcnn[2])) is OnnxRuntimeTorchvisionDetection


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

    n_strict = 0
    for image in images:
        default_predictions, strict_predictions = default_model.predict(image), strict_model.predict(image)
        assert len(strict_predictions) < len(default_predictions)
        assert strict_predictions == [p for p in default_predictions if p.confidence > 0.5]
        n_strict += len(strict_predictions)
    assert n_strict > 0


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
