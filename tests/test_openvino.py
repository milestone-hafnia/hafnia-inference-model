"""End-to-end: ONNX bundle -> OpenVINO (ONNX and IR bundles), checked against ONNX Runtime.

The ONNX Runtime flavors are the reference (they are checked against torch in
'test_torchvision_onnxruntime.py'). These cross-engine parity tests also keep the duplicated
postprocessing of the two engine packages in line.
"""

import math
import zipfile

import numpy as np
import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark
from helpers import assert_detections_match

from inference_model_core import inference_model_from_path, read_model_info

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("openvino")
from inference_model_onnxruntime.flavors import OnnxRuntimeTorchvisionSSD
from inference_model_openvino.export_onnx import export_onnx_bundle_to_openvino
from inference_model_openvino.flavors import OpenVINOTorchvisionDetection, OpenVINOTorchvisionSSD

pytestmark = pytest.mark.slow


def _convert(tmp_path_factory, path_onnx_bundle, precision):
    path_bundle = tmp_path_factory.mktemp("openvino") / f"{path_onnx_bundle.stem}_openvino_{precision}.zip"
    return export_onnx_bundle_to_openvino(path_onnx_bundle, path_bundle, precision=precision)


@pytest.fixture(scope="module")
def ssd_ir(ssd, tmp_path_factory):
    return _convert(tmp_path_factory, ssd[2], "fp32")


@pytest.fixture(scope="module")
def ssd_ir_fp16(ssd, tmp_path_factory):
    return _convert(tmp_path_factory, ssd[2], "fp16")


@pytest.fixture(scope="module")
def frcnn_ir(frcnn, tmp_path_factory):
    return _convert(tmp_path_factory, frcnn[2], "fp32")


def test_flavor_selection(ssd, frcnn, ssd_ir, frcnn_ir):
    # ONNX bundles run on OpenVINO through the engine override, IR bundles default to OpenVINO
    assert type(inference_model_from_path(ssd[2])) is OnnxRuntimeTorchvisionSSD
    assert type(inference_model_from_path(ssd[2], inference_engine="openvino")) is OpenVINOTorchvisionSSD
    assert type(inference_model_from_path(ssd_ir)) is OpenVINOTorchvisionSSD
    assert type(inference_model_from_path(frcnn_ir)) is OpenVINOTorchvisionDetection
    with pytest.raises(ValueError, match="can not run model format 'openvino'"):
        inference_model_from_path(ssd_ir, inference_engine="onnxruntime")


def test_ir_model_info(ssd, ssd_ir, ssd_ir_fp16):
    onnx_info, ir_info, fp16_info = read_model_info(ssd[2]), read_model_info(ssd_ir), read_model_info(ssd_ir_fp16)

    assert ir_info.runtime.model_format == "openvino"
    assert ir_info.runtime.inference_engine == "openvino"
    assert ir_info.runtime.all_files() == ["model.xml", "model.bin"]
    assert (ir_info.runtime.precision, fp16_info.runtime.precision) == ("fp32", "fp16")
    assert ir_info.runtime.provenance["compress_to_fp16"] is False
    assert ir_info.runtime.provenance["parent"]["provenance"] == onnx_info.runtime.provenance

    # Everything that defines how the model is used is carried over unchanged
    assert ir_info.tasks == onnx_info.tasks
    for field in ("input", "outputs", "preprocessing", "postprocessing"):
        assert getattr(ir_info.runtime, field) == getattr(onnx_info.runtime, field)

    # FP32 keeps the weights as is, FP16 halves them (design review 4.2)
    with zipfile.ZipFile(ssd_ir) as fp32_zip, zipfile.ZipFile(ssd_ir_fp16) as fp16_zip:
        fp32_size, fp16_size = fp32_zip.getinfo("model.bin").file_size, fp16_zip.getinfo("model.bin").file_size
    assert 0.45 < fp16_size / fp32_size < 0.55


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
def test_frcnn_blank_images_match_onnxruntime(frcnn, frcnn_ir, fill_value):
    """Blank images take the zero-proposals path of the original model. The exported graph runs it without
    the original's score filter (see the RPN score threshold in the exporter), so both engines must agree."""
    image = np.full((300, 400, 3), fill_value, dtype=np.uint8)
    expected = inference_model_from_path(frcnn[2]).predict(image)
    predicted = inference_model_from_path(frcnn_ir).predict(image)
    settings = {"min_score": 0.1, "min_iou": 0.99, "max_score_diff": 1e-3}
    assert_detections_match(expected, predicted, **settings)
    assert_detections_match(predicted, expected, **settings)


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


def test_converter_needs_an_onnx_bundle(ssd_ir, tmp_path):
    with pytest.raises(ValueError, match="Expected an ONNX bundle"):
        export_onnx_bundle_to_openvino(ssd_ir, tmp_path / "again.zip")


def test_hafnia_benchmark_on_micro_coco(ssd, ssd_ir, micro_coco):
    """Same metrics on both engines through hafnia's benchmark."""
    onnxruntime_metrics, _ = run_benchmark(dataset=micro_coco, model=inference_model_from_path(ssd[2]), batch_size=2)
    openvino_metrics, _ = run_benchmark(dataset=micro_coco, model=inference_model_from_path(ssd_ir), batch_size=2)

    assert openvino_metrics.keys() == onnxruntime_metrics.keys()
    assert all(math.isfinite(value) for value in openvino_metrics.values())
    for name, value in onnxruntime_metrics.items():
        assert openvino_metrics[name] == pytest.approx(value, abs=1e-3), name
