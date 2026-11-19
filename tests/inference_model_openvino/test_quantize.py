"""INT8 post-training quantization (NNCF) of FP32 bundles, calibrated on the micro COCO images.

Calibrating and evaluating on the same 8 images only shows that quantization works and keeps the model
close to FP32. Real accuracy numbers need a larger dataset.
"""

import contextlib
import io
import zipfile

import numpy as np
import pytest
from hafnia.dataset.benchmark.benchmark import run_benchmark

from inference_model_core import inference_model_from_path, read_model_info

pytest.importorskip("torch")
pytest.importorskip("nncf")
from inference_model_openvino.flavors import OpenVINOTorchvisionDetection, OpenVINOTorchvisionSSD
from inference_model_openvino.quantize import quantize_bundle

pytestmark = pytest.mark.slow

# INT8 mAP may be this much lower than FP32 on the micro dataset. Quantizing the SSD box decoding (no
# quantization hints) dropped mAP by 0.2, so this catches graph parts that must stay in floating point.
MAX_MAP_DROP = 0.1


@pytest.fixture(scope="module")
def ssd_int8(ssd, images, tmp_path_factory):
    return quantize_bundle(ssd[2], tmp_path_factory.mktemp("int8") / "ssd_int8.zip", images)


@pytest.fixture(scope="module")
def frcnn_int8(frcnn, images, tmp_path_factory):
    return quantize_bundle(frcnn[2], tmp_path_factory.mktemp("int8") / "frcnn_int8.zip", images)


def mean_average_precision(path_bundle, dataset) -> float:
    with contextlib.redirect_stdout(io.StringIO()):  # Hide the COCO evaluation report
        metrics, _ = run_benchmark(dataset=dataset, model=inference_model_from_path(path_bundle))
    return metrics["object_detection/mAP"]


def test_int8_model_info(ssd, ssd_ir, ssd_int8, images):
    onnx_info, int8_info = read_model_info(ssd[2]), read_model_info(ssd_int8)

    assert (int8_info.runtime.model_format, int8_info.runtime.precision) == ("openvino", "int8")
    quantization = int8_info.runtime.provenance["quantization"]
    assert quantization["preset"] == "mixed"
    assert quantization["calibration_samples"] == len(images)
    assert quantization["ignored_node_patterns"] == onnx_info.runtime.quantization_hints.ignored_node_patterns
    assert int8_info.runtime.provenance["parent"]["precision"] == "fp32"

    # Everything that defines how the model is used is carried over unchanged
    assert int8_info.tasks == onnx_info.tasks
    for field in ("input", "outputs", "preprocessing", "postprocessing", "quantization_hints"):
        assert getattr(int8_info.runtime, field) == getattr(onnx_info.runtime, field)

    # INT8 weights are about a quarter of FP32. Graph parts kept in floating point add a little.
    with zipfile.ZipFile(ssd_ir) as fp32_zip, zipfile.ZipFile(ssd_int8) as int8_zip:
        fp32_size, int8_size = fp32_zip.getinfo("model.bin").file_size, int8_zip.getinfo("model.bin").file_size
    assert int8_size / fp32_size < 0.4


@pytest.mark.parametrize("model_fixture", ["ssd", "frcnn"])
def test_int8_accuracy_close_to_fp32(model_fixture, request, micro_coco):
    path_fp32 = request.getfixturevalue(f"{model_fixture}_ir")
    path_int8 = request.getfixturevalue(f"{model_fixture}_int8")
    expected_flavor = {"ssd": OpenVINOTorchvisionSSD, "frcnn": OpenVINOTorchvisionDetection}[model_fixture]

    assert type(inference_model_from_path(path_int8)) is expected_flavor
    fp32_map, int8_map = mean_average_precision(path_fp32, micro_coco), mean_average_precision(path_int8, micro_coco)
    assert int8_map >= fp32_map - MAX_MAP_DROP, f"INT8 mAP {int8_map:.3f}, FP32 mAP {fp32_map:.3f}"


def test_frcnn_int8_images_without_proposals(frcnn_int8):
    model = inference_model_from_path(frcnn_int8)
    assert model.predict(np.zeros((300, 400, 3), dtype=np.uint8)) == []


def test_quantize_openvino_ir_bundle(ssd_ir, images, tmp_path):
    path_int8 = quantize_bundle(ssd_ir, tmp_path / "from_ir.zip", images[:2])
    assert read_model_info(path_int8).runtime.provenance["parent"]["model_format"] == "openvino"
    assert inference_model_from_path(path_int8).predict(images[0])


def test_quantize_needs_fp32_bundle_and_images(ssd, ssd_ir_fp16, images, tmp_path):
    with pytest.raises(ValueError, match="Expected an FP32 bundle"):
        quantize_bundle(ssd_ir_fp16, tmp_path / "int8.zip", images)
    with pytest.raises(ValueError, match="At least one calibration image"):
        quantize_bundle(ssd[2], tmp_path / "int8.zip", [])


def test_resnet18_int8_agrees_with_fp32(resnet18, resnet18_ir, images, tmp_path):
    path_int8 = quantize_bundle(resnet18[2], tmp_path / "resnet18_int8.zip", images)
    fp32_model, int8_model = inference_model_from_path(resnet18_ir), inference_model_from_path(path_int8)
    n_same = sum(fp32_model.predict(image)[0].class_idx == int8_model.predict(image)[0].class_idx for image in images)
    assert n_same >= len(images) - 2


def test_rfdetr_int8_accuracy_close_to_fp32(rfdetr, rfdetr_ir, images, micro_coco, tmp_path):
    """RF-DETR is quantized in NNCF's transformer mode (quantization hint). The default dropped mAP to 0."""
    path_int8 = quantize_bundle(rfdetr[2], tmp_path / "rfdetr_int8.zip", images)
    assert read_model_info(path_int8).runtime.provenance["quantization"]["model_type"] == "transformer"
    fp32_map, int8_map = mean_average_precision(rfdetr_ir, micro_coco), mean_average_precision(path_int8, micro_coco)
    assert int8_map >= fp32_map - MAX_MAP_DROP, f"INT8 mAP {int8_map:.3f}, FP32 mAP {fp32_map:.3f}"
