"""ONNX bundle -> OpenVINO IR bundle conversion: model info, precision and input checks."""

import zipfile

import pytest

from inference_model_core import read_model_info

pytest.importorskip("torch")
pytest.importorskip("openvino")
from inference_model_openvino.export_onnx import export_onnx_bundle_to_openvino

pytestmark = pytest.mark.slow


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


def test_converter_needs_an_onnx_bundle(ssd_ir, tmp_path):
    with pytest.raises(ValueError, match="Expected an ONNX bundle"):
        export_onnx_bundle_to_openvino(ssd_ir, tmp_path / "again.zip")
