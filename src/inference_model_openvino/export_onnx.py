"""Convert an ONNX model bundle to an OpenVINO IR bundle ('model.xml' + 'model.bin').

Generic over model families: the graph is converted as is and the model info is carried over, with only
the runtime fields that describe the files changed. Preprocessing and decoder settings stay the same, so
the IR bundle runs with the same flavor family as the ONNX bundle.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import openvino as ov

from inference_model_core import ModelBundleInfo, open_bundle, write_bundle
from inference_model_core.inference_model import PathLike

ENGINE = "openvino"
MODEL_FORMAT = "openvino"
IR_MODEL_FILENAME = "model.xml"
IR_WEIGHTS_FILENAME = "model.bin"

IRPrecision = Literal["fp32", "fp16"]


def export_onnx_bundle_to_openvino(
    path_onnx_bundle: PathLike,
    path_bundle: PathLike,
    precision: IRPrecision = "fp32",
    overwrite: bool = False,
) -> Path:
    """Convert an ONNX bundle to an OpenVINO IR bundle.

    Args:
        path_onnx_bundle: ONNX bundle (zip or directory).
        path_bundle: Output bundle. A path ending with '.zip' creates a zip bundle, otherwise a directory.
        precision: Weight precision of the IR. 'fp32' keeps the ONNX weights. 'fp16' stores weights as FP16,
            which halves the file size. Devices still choose their own compute precision.
        overwrite: Replace an existing output bundle.
    """
    with open_bundle(path_onnx_bundle) as path_dir:
        model_info = ModelBundleInfo.from_json_file(path_dir / "model_info.json")
        if model_info.runtime.model_format != "onnx":
            raise ValueError(f"Expected an ONNX bundle, got model format '{model_info.runtime.model_format}'.")
        model = ov.convert_model(path_dir / model_info.runtime.file_path)

    return write_openvino_bundle(
        model,
        parent_info=model_info,
        path_bundle=path_bundle,
        precision=precision,
        provenance={"conversion": {"tool": "openvino.convert_model", "source_format": "onnx"}},
        overwrite=overwrite,
    )


def write_openvino_bundle(
    model: ov.Model,
    parent_info: ModelBundleInfo,
    path_bundle: PathLike,
    precision: Literal["fp32", "fp16", "int8"],
    provenance: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Write an OpenVINO model as an IR bundle, with the model info of the bundle it was derived from.

    The parent's runtime description (format, files, precision, provenance) is kept under
    'provenance.parent', so a bundle can be traced back to its source.

    Args:
        model: Converted (or quantized) OpenVINO model.
        parent_info: Model info of the source bundle.
        path_bundle: Output bundle path.
        precision: Recorded precision. Weights are compressed to FP16 only for 'fp16'. 'ov.save_model'
            compresses by default, which would silently turn an FP32 model into FP16 (design review 4.2).
        provenance: Extra provenance, e.g. conversion or quantization settings.
        overwrite: Replace an existing output bundle.
    """
    parent_runtime = parent_info.runtime
    runtime = parent_runtime.model_copy(
        update={
            "model_format": MODEL_FORMAT,
            "inference_engine": ENGINE,
            "file_path": IR_MODEL_FILENAME,
            "extra_files": [IR_WEIGHTS_FILENAME],
            "precision": precision,
            "provenance": {
                **(provenance or {}),
                "openvino_version": ov.get_version(),
                "compress_to_fp16": precision == "fp16",
                "parent": {
                    "model_format": parent_runtime.model_format,
                    "file_path": parent_runtime.file_path,
                    "precision": parent_runtime.precision,
                    "provenance": parent_runtime.provenance,
                },
            },
        }
    )
    model_info = parent_info.model_copy(update={"runtime": runtime})

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_xml = Path(tmp_dir) / IR_MODEL_FILENAME
        ov.save_model(model, path_xml, compress_to_fp16=precision == "fp16")
        files = {
            IR_MODEL_FILENAME: path_xml.read_bytes(),
            IR_WEIGHTS_FILENAME: (Path(tmp_dir) / IR_WEIGHTS_FILENAME).read_bytes(),
        }
    return write_bundle(path_bundle, model_info, files, overwrite=overwrite)
