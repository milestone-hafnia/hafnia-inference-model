"""OpenVINO flavors on ONNX and IR bundles, checked against ONNX Runtime.

The ONNX Runtime flavors are the reference (they are checked against torch in 'tests/model_exporters').
These cross-engine parity tests also keep the duplicated postprocessing of the two engine packages in line.
"""

import math

import numpy as np
import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark
from helpers import assert_detections_match

from inference_model_core import inference_model_from_path

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("openvino")
from inference_model_onnxruntime.flavors import OnnxRuntimeTorchvisionSSD
from inference_model_openvino.flavors import OpenVINOTorchvisionDetection, OpenVINOTorchvisionSSD

pytestmark = pytest.mark.slow


def test_flavor_selection(ssd, frcnn, ssd_ir, frcnn_ir):
    # ONNX bundles run on OpenVINO through the engine override, IR bundles default to OpenVINO
    assert type(inference_model_from_path(ssd[2])) is OnnxRuntimeTorchvisionSSD
    assert type(inference_model_from_path(ssd[2], inference_engine="openvino")) is OpenVINOTorchvisionSSD
    assert type(inference_model_from_path(ssd_ir)) is OpenVINOTorchvisionSSD
    assert type(inference_model_from_path(frcnn_ir)) is OpenVINOTorchvisionDetection
    with pytest.raises(ValueError, match="can not run model format 'openvino'"):
        inference_model_from_path(ssd_ir, inference_engine="onnxruntime")


def test_ssd_raw_outputs_match_onnxruntime(ssd, ssd_ir, images):
    reference = inference_model_from_path(ssd[2])
    models = [inference_model_from_path(ssd[2], inference_engine="openvino"), inference_model_from_path(ssd_ir)]
    for image in images:
        inputs, _ = reference.preprocess(image)
        reference_outputs = reference.predict_raw(inputs)
        for model in models:
            outputs = model.predict_raw(inputs)
            assert outputs.keys() == reference_outputs.keys()
            np.testing.assert_allclose(outputs["boxes"], reference_outputs["boxes"], atol=1e-2)  # Input pixels
            np.testing.assert_allclose(outputs["scores"], reference_outputs["scores"], atol=1e-4)


@pytest.mark.parametrize("model_fixture", ["ssd", "frcnn"])
def test_predictions_match_onnxruntime(model_fixture, request, tmp_path_factory, images):
    """Same primitives from ONNX Runtime and OpenVINO, in both directions (nothing missing, nothing extra)."""
    path_onnx = request.getfixturevalue(model_fixture)[2]
    path_ir = request.getfixturevalue(f"{model_fixture}_ir")
    reference = inference_model_from_path(path_onnx)
    model = inference_model_from_path(path_ir)
    for image in images:
        expected, predicted = reference.predict(image), model.predict(image)
        assert expected
        settings = {"min_score": 0.1, "min_iou": 0.99, "max_score_diff": 1e-3}
        assert_detections_match(expected, predicted, **settings)
        assert_detections_match(predicted, expected, **settings)


def test_fp16_predictions_close_to_fp32(ssd_ir, ssd_ir_fp16, images):
    fp32_model, fp16_model = inference_model_from_path(ssd_ir), inference_model_from_path(ssd_ir_fp16)
    n_checked = sum(
        assert_detections_match(
            fp32_model.predict(image), fp16_model.predict(image), min_score=0.3, min_iou=0.97, max_score_diff=0.02
        )
        for image in images
    )
    assert n_checked >= len(images)


def test_ssd_batching(ssd_ir, images):
    model = inference_model_from_path(ssd_ir)
    assert model.supports_batching()
    for image, predictions in zip(images, model.predict_batch(images), strict=True):
        assert_detections_match(model.predict(image), predictions, min_score=0.01, min_iou=0.999, max_score_diff=1e-4)


@pytest.mark.parametrize("fill_value", [0, 128, 255])
def test_frcnn_images_without_proposals(frcnn, frcnn_ir, fill_value):
    """Blank images take the zero-boxes path of the graph. OpenVINO used to return one row of uninitialized
    memory there (see '_batched_nms_export' in the exporter): the outputs must be empty on both formats."""
    image = np.full((300, 400, 3), fill_value, dtype=np.uint8)
    onnx_on_openvino = inference_model_from_path(frcnn[2], inference_engine="openvino")
    for model in (onnx_on_openvino, inference_model_from_path(frcnn_ir)):
        raw_outputs = model.predict_raw(model.preprocess(image)[0])
        assert [output.shape[0] for output in raw_outputs.values()] == [0, 0, 0]
        assert model.predict(image) == []


def test_write_model_round_trip(ssd_ir, images, tmp_path):
    model = inference_model_from_path(ssd_ir)
    path_copy = model.write_model(tmp_path / "copy")
    assert sorted(p.name for p in path_copy.iterdir()) == ["model.bin", "model.xml", "model_info.json"]
    assert inference_model_from_path(path_copy).predict(images[0]) == model.predict(images[0])


def test_engine_options(ssd_ir, images):
    model = inference_model_from_path(ssd_ir, device="CPU", num_threads=1, performance_hint="latency")
    assert model.predict(images[0])
    with pytest.raises(ValueError, match="Unsupported performance hint"):
        inference_model_from_path(ssd_ir, performance_hint="fastest")
    with pytest.raises(TypeError, match="Unknown OpenVINO engine options"):
        inference_model_from_path(ssd_ir, unknown_option=1)


def test_hafnia_benchmark_on_micro_coco(ssd, ssd_ir, micro_coco):
    """Same metrics on both engines through hafnia's benchmark."""
    onnxruntime_metrics, _ = run_benchmark(dataset=micro_coco, model=inference_model_from_path(ssd[2]), batch_size=2)
    openvino_metrics, _ = run_benchmark(dataset=micro_coco, model=inference_model_from_path(ssd_ir), batch_size=2)

    assert openvino_metrics.keys() == onnxruntime_metrics.keys()
    assert all(math.isfinite(value) for value in openvino_metrics.values())
    for name, value in onnxruntime_metrics.items():
        assert openvino_metrics[name] == pytest.approx(value, abs=1e-3), name


def test_classification_matches_onnxruntime(resnet18, resnet18_ir, images):
    from inference_model_core.classification import softmax
    from inference_model_openvino.flavors import OpenVINOTorchvisionClassification

    reference, model = inference_model_from_path(resnet18[2]), inference_model_from_path(resnet18_ir)
    assert type(model) is OpenVINOTorchvisionClassification
    for image in images:
        inputs, _ = reference.preprocess(image)
        expected = softmax(reference.predict_raw(inputs)["logits"])
        np.testing.assert_allclose(softmax(model.predict_raw(inputs)["logits"]), expected, atol=1e-4)
        assert model.predict(image)[0].class_idx == reference.predict(image)[0].class_idx


def test_rfdetr_matches_onnxruntime(rfdetr, rfdetr_ir, images):
    """OpenVINO differs from ONNX Runtime in low-score queries of RF-DETR's transformer (boxes moved by up to 0.4 of
    the image on the micro dataset), so only detections with score >= 0.1 are compared, in both directions."""
    from inference_model_openvino.flavors import OpenVINORFDETRDetection

    reference, model = inference_model_from_path(rfdetr[2]), inference_model_from_path(rfdetr_ir)
    assert type(model) is OpenVINORFDETRDetection
    settings = {"min_score": 0.1, "min_iou": 0.99, "max_score_diff": 1e-2}
    for image in images:
        expected, predicted = reference.predict(image), model.predict(image)
        assert_detections_match(expected, predicted, **settings)
        assert_detections_match(predicted, expected, **settings)
