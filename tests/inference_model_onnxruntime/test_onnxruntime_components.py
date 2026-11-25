"""The ONNX Runtime network component on ONNX bundles of torchvision and RF-DETR models: batching, runtime
settings, storage and Hafnia's benchmark. Parity with torch is tested with the bundle helpers
('tests/model_exporters')."""

import json
import math

import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark
from helpers import assert_detections_match, network_of

from inference_model_core import AlignPredictionClasses, GraphInferenceModel, ModelBundle

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
from inference_model_onnxruntime.components import OnnxRuntimeNetwork

pytestmark = pytest.mark.slow


def test_onnx_networks_run_on_onnxruntime_by_default(ssd, frcnn):
    for path_bundle in (ssd[2], frcnn[2]):
        model = GraphInferenceModel.from_path(path_bundle)
        assert type(network_of(model)) is OnnxRuntimeNetwork
        # Everything else is written in torch
        assert {type(component).engine for component in model.graph.components} == {"onnxruntime", "torch"}


def test_ssd_batching(ssd, images):
    model = GraphInferenceModel.from_path(ssd[2])
    assert network_of(model).info.batched and network_of(model).max_batch_size is None

    batch_predictions = model.predict_batch(images)
    for image, predictions in zip(images, batch_predictions, strict=True):
        assert_detections_match(model.predict(image), predictions, min_score=0.01, min_iou=0.999, max_score_diff=1e-4)


def test_frcnn_runs_image_by_image(frcnn, images):
    model = GraphInferenceModel.from_path(frcnn[2])
    assert not network_of(model).info.batched
    assert len(model.predict_batch(images)) == len(images)


def test_score_threshold_is_a_runtime_setting(ssd, images):
    default_model = GraphInferenceModel.from_path(ssd[2])
    strict_model = GraphInferenceModel.from_path(ssd[2], params={"score_threshold": 0.5})

    n_strict = 0
    for image in images:
        default_predictions, strict_predictions = default_model.predict(image), strict_model.predict(image)
        assert len(strict_predictions) < len(default_predictions)
        assert strict_predictions == [p for p in default_predictions if p.confidence > 0.5]
        n_strict += len(strict_predictions)
    assert n_strict > 0


def test_bundle_copy_round_trip(ssd, images, tmp_path):
    model = GraphInferenceModel.from_path(ssd[2])
    path_copy = model.get_model_info().write(tmp_path / "copy").path
    assert sorted(p.name for p in (path_copy / "network").iterdir()) == ["model.onnx"]
    assert GraphInferenceModel.from_path(path_copy).predict(images[0]) == model.predict(images[0])


def test_engine_options(ssd):
    with pytest.raises(ValueError, match="Unsupported device"):
        GraphInferenceModel.from_path(ssd[2], device="TPU")
    with pytest.raises(TypeError, match="Unknown ONNX Runtime engine options"):
        GraphInferenceModel.from_path(ssd[2], engine_options={"onnxruntime": {"unknown_option": 1}})
    assert GraphInferenceModel.from_path(ssd[2], num_threads=1)


def test_model_info_must_describe_the_onnx_graph(ssd, tmp_path):
    bundle = ModelBundle.from_path(ssd[2]).write(tmp_path / "bundle")
    path_bundle = bundle.path
    network = bundle.graph.get_component("network")
    network.params["output_tensors"][0]["name"] = "detections"
    network.outputs["detections"] = network.outputs.pop("boxes")
    (path_bundle / "model_info.json").write_text(json.dumps(bundle.model_dump(mode="json")))

    with pytest.raises(ValueError, match="does not match its model file"):
        GraphInferenceModel.from_path(path_bundle)


def test_hafnia_benchmark_on_micro_coco(ssd, micro_coco):
    model = GraphInferenceModel.from_path(ssd[2])
    recipe = AlignPredictionClasses()
    metrics, dataset_predictions = run_benchmark(micro_coco, model, recipe_transforms=[recipe], batch_size=2)

    assert "object_detection/predictions" in [task.name for task in dataset_predictions.info.tasks]
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())
    assert max(metrics.values()) > 0


def test_classification_and_rfdetr_with_batching(resnet18, rfdetr, images):
    """Both networks have a dynamic batch axis: a batched call gives the same predictions as one call per image."""
    pytest.importorskip("rfdetr")
    classifier, detector = GraphInferenceModel.from_path(resnet18[2]), GraphInferenceModel.from_path(rfdetr[2])
    for model in (classifier, detector):
        assert type(network_of(model)) is OnnxRuntimeNetwork and network_of(model).info.batched

    for image, (prediction,) in zip(images, classifier.predict_batch(images), strict=True):
        (single,) = classifier.predict(image)
        assert prediction.class_idx == single.class_idx
        assert prediction.confidence == pytest.approx(single.confidence, abs=1e-5)
    for image, predictions in zip(images, detector.predict_batch(images), strict=True):
        settings = {"min_score": 0.05, "min_iou": 0.999, "max_score_diff": 1e-4}
        assert_detections_match(detector.predict(image), predictions, **settings)
