"""Conversion of ONNX bundles to OpenVINO IR bundles: FP32, FP16 and INT8 post-training quantization (NNCF).

INT8 is calibrated on the micro COCO images. Calibrating and evaluating on the same 8 images only shows that
quantization works and keeps the model close to FP32. Real accuracy numbers need a larger dataset.
"""

import contextlib
import io
import zipfile

import numpy as np
import pytest
from helpers import network_of

from inference_model_core import GraphInferenceModel, ModelBundle, NetworkParams

pytest.importorskip("torch")
pytest.importorskip("nncf")
from inference_model_openvino.components import OpenVINOIRNetwork

pytestmark = pytest.mark.slow

# INT8 mAP may be this much lower than FP32 on the micro dataset. Quantizing the SSD box decoding (no
# 'ignored_node_patterns') dropped mAP by 0.2, so this catches graph parts that must stay in floating point.
MAX_MAP_DROP = 0.1
IR_FILES = ["network/model.xml", "network/model.bin"]


def quantize(path_source, path_bundle, images, **kwargs):
    calibration_inputs = [{"image": image} for image in images]
    source = ModelBundle.from_path(path_source)
    return source.convert("openvino_int8", path_bundle, calibration_inputs=calibration_inputs, **kwargs).path


@pytest.fixture(scope="module")
def ssd_int8(ssd, images, tmp_path_factory):
    return quantize(ssd[2], tmp_path_factory.mktemp("int8") / "ssd_int8.zip", images)


@pytest.fixture(scope="module")
def frcnn_int8(frcnn, images, tmp_path_factory):
    return quantize(frcnn[2], tmp_path_factory.mktemp("int8") / "frcnn_int8.zip", images)


def network_info(path_bundle):
    network = ModelBundle.from_path(path_bundle).graph.get_component("network")
    return network, NetworkParams.model_validate(network.params)


def weights_size(path_bundle) -> int:
    with zipfile.ZipFile(path_bundle) as archive:
        return archive.getinfo("network/model.bin").file_size


def mean_average_precision(path_bundle, dataset) -> float:
    with contextlib.redirect_stdout(io.StringIO()):  # Hide the COCO evaluation report
        metrics, _ = GraphInferenceModel.from_path(path_bundle).benchmark(dataset)
    return metrics["object_detection/mAP"]


def test_ir_model_info(ssd, ssd_ir, ssd_ir_fp16):
    onnx_info, ir_info, fp16_info = (
        ModelBundle.from_path(ssd[2]),
        ModelBundle.from_path(ssd_ir),
        ModelBundle.from_path(ssd_ir_fp16),
    )
    (onnx_network, onnx_params), (ir_network, ir_params) = network_info(ssd[2]), network_info(ssd_ir)
    fp16_network, fp16_params = network_info(ssd_ir_fp16)

    assert (onnx_network.type, ir_network.type, ir_network.files) == ("onnx_network", "openvino_network", IR_FILES)
    assert (ir_params.precision, fp16_params.precision) == ("fp32", "fp16")
    assert ir_network.provenance["compress_to_fp16"] is False
    assert ir_network.provenance["parent"]["provenance"] == onnx_network.provenance
    assert ir_network.provenance["parent"]["type"] == "onnx_network"
    # An FP32 IR can still be quantized, an FP16 IR can not be converted further
    assert (sorted(ir_network.conversions), fp16_network.conversions) == (["openvino_int8"], {})
    assert ir_network.conversions["openvino_int8"] == onnx_network.conversions["openvino_int8"]

    # Everything that defines how the model is used is carried over unchanged
    assert ir_info.tasks == onnx_info.tasks
    assert (ir_network.inputs, ir_network.outputs, ir_network.batched) == (
        onnx_network.inputs,
        onnx_network.outputs,
        onnx_network.batched,
    )
    assert (ir_params.input_tensors, ir_params.output_tensors) == (
        onnx_params.input_tensors,
        onnx_params.output_tensors,
    )
    for component in onnx_info.graph.components:
        if component.name != "network":
            assert ir_info.graph.get_component(component.name) == component
            assert fp16_info.graph.get_component(component.name) == component

    # FP32 keeps the weights as is, FP16 halves them
    assert 0.45 < weights_size(ssd_ir_fp16) / weights_size(ssd_ir) < 0.55


def test_conversions_that_are_not_possible(ssd, ssd_ir, ssd_ir_fp16, images, tmp_path):
    path_bundle = tmp_path / "bundle.zip"
    with pytest.raises(ValueError, match="can be converted to 'openvino'"):
        ModelBundle.from_path(ssd_ir).convert("openvino", path_bundle)  # Already OpenVINO IR
    with pytest.raises(ValueError, match="can be converted to 'openvino_int8'"):
        quantize(ssd_ir_fp16, path_bundle, images)  # Only FP32 networks are quantized
    with pytest.raises(ValueError, match="needs calibration inputs"):
        ModelBundle.from_path(ssd[2]).convert("openvino_int8", path_bundle)
    with pytest.raises(ValueError, match="'precision' must be 'fp32' or 'fp16'"):
        ModelBundle.from_path(ssd[2]).convert("openvino", path_bundle, options={"precision": "int8"})
    with pytest.raises(ValueError, match=r"Unknown options \['presets'\] for target 'openvino_int8'"):
        quantize(ssd[2], path_bundle, images, options={"presets": "mixed"})
    assert not path_bundle.exists()


def test_int8_model_info(ssd, ssd_ir, ssd_int8, images):
    onnx_info, int8_info = ModelBundle.from_path(ssd[2]), ModelBundle.from_path(ssd_int8)
    (onnx_network, _), (int8_network, int8_params) = network_info(ssd[2]), network_info(ssd_int8)

    assert (int8_network.type, int8_network.files, int8_params.precision) == ("openvino_network", IR_FILES, "int8")
    assert int8_network.conversions == {}
    quantization = int8_network.provenance["quantization"]
    assert quantization["preset"] == "mixed"
    assert quantization["calibration_samples"] == len(images)
    # Which nodes stay in floating point comes from the conversion options stored with the network
    stored_options = onnx_network.conversions["openvino_int8"].options
    assert quantization["ignored_node_patterns"] == stored_options["ignored_node_patterns"] != []
    assert int8_network.provenance["parent"]["precision"] == "fp32"

    # Everything that defines how the model is used is carried over unchanged
    assert int8_info.tasks == onnx_info.tasks
    for component in onnx_info.graph.components:
        if component.name != "network":
            assert int8_info.graph.get_component(component.name) == component

    # INT8 weights are about a quarter of FP32. Graph parts kept in floating point add a little.
    assert weights_size(ssd_int8) / weights_size(ssd_ir) < 0.4


@pytest.mark.parametrize("model_fixture", ["ssd", "frcnn"])
def test_int8_accuracy_close_to_fp32(model_fixture, request, micro_coco):
    path_fp32 = request.getfixturevalue(f"{model_fixture}_ir")
    path_int8 = request.getfixturevalue(f"{model_fixture}_int8")

    assert type(network_of(GraphInferenceModel.from_path(path_int8))) is OpenVINOIRNetwork
    fp32_map, int8_map = mean_average_precision(path_fp32, micro_coco), mean_average_precision(path_int8, micro_coco)
    assert int8_map >= fp32_map - MAX_MAP_DROP, f"INT8 mAP {int8_map:.3f}, FP32 mAP {fp32_map:.3f}"


def test_frcnn_int8_images_without_proposals(frcnn_int8):
    model = GraphInferenceModel.from_path(frcnn_int8)
    assert model.predict(np.zeros((300, 400, 3), dtype=np.uint8)) == []


def test_quantize_openvino_ir_bundle(ssd_ir, images, tmp_path):
    provenance = {"calibration_dataset": "micro-coco-2017"}
    options = {"preset": "performance"}
    path_int8 = quantize(ssd_ir, tmp_path / "from_ir.zip", images[:2], options=options, provenance=provenance)
    network, _ = network_info(path_int8)

    assert network.provenance["parent"]["type"] == "openvino_network"
    assert network.provenance["calibration_dataset"] == "micro-coco-2017"
    assert (
        network.provenance["quantization"]["preset"],
        network.provenance["quantization"]["calibration_samples"],
    ) == (
        "performance",
        2,
    )
    assert GraphInferenceModel.from_path(path_int8).predict(images[0])


def test_resnet18_int8_agrees_with_fp32(resnet18, resnet18_ir, images, tmp_path):
    path_int8 = quantize(resnet18[2], tmp_path / "resnet18_int8.zip", images)
    fp32_model, int8_model = GraphInferenceModel.from_path(resnet18_ir), GraphInferenceModel.from_path(path_int8)
    n_same = sum(fp32_model.predict(image)[0].class_idx == int8_model.predict(image)[0].class_idx for image in images)
    assert n_same >= len(images) - 2


def test_rfdetr_int8_accuracy_close_to_fp32(rfdetr, rfdetr_ir, images, micro_coco, tmp_path):
    """RF-DETR is quantized in NNCF's transformer mode (a stored conversion option). The default dropped mAP to 0."""
    path_int8 = quantize(rfdetr[2], tmp_path / "rfdetr_int8.zip", images)
    assert network_info(path_int8)[0].provenance["quantization"]["model_type"] == "transformer"
    fp32_map, int8_map = mean_average_precision(rfdetr_ir, micro_coco), mean_average_precision(path_int8, micro_coco)
    assert int8_map >= fp32_map - MAX_MAP_DROP, f"INT8 mAP {int8_map:.3f}, FP32 mAP {fp32_map:.3f}"
