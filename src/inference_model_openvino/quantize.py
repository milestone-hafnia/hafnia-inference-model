"""INT8 post-training quantization (NNCF) of ONNX or OpenVINO FP32 bundles into OpenVINO IR bundles.

Generic over model families:

- Calibration images are preprocessed with 'preprocess_model_input', the same model info driven function
  that runs at inference, so the model is calibrated on exactly the inputs it will see.
- Graph parts that should stay in floating point (e.g. box decoding in the graph) come from the model
  info's 'quantization_hints', written by the exporter that built the graph.
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import nncf
import openvino as ov
from hafnia.dataset.benchmark.inference_model import ImageType

from inference_model_core import ModelBundleInfo, open_bundle
from inference_model_core.inference_model import PathLike
from inference_model_core.preprocessing import preprocess_model_input
from inference_model_openvino.export_onnx import write_openvino_bundle

QUANTIZABLE_FORMATS = ("onnx", "openvino")
# Fewer calibration images work, but NNCF advises at least 300 for reliable activation ranges
RECOMMENDED_CALIBRATION_SAMPLES = 300

QuantizationPreset = Literal["mixed", "performance"]


def quantize_bundle(
    path_source_bundle: PathLike,
    path_bundle: PathLike,
    calibration_images: Sequence[ImageType],
    preset: QuantizationPreset = "mixed",
    provenance: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Quantize the weights and activations of an FP32 bundle to INT8 and write an OpenVINO IR bundle.

    Args:
        path_source_bundle: FP32 ONNX or OpenVINO IR bundle (zip or directory).
        path_bundle: Output bundle. A path ending with '.zip' creates a zip bundle, otherwise a directory.
        calibration_images: Images (arrays, file paths or PIL images) representative of the deployment data.
            They are loaded one at a time, so file paths keep memory use low.
        preset: NNCF quantization preset. 'mixed' quantizes activations asymmetrically, which suits
            activations that are not centered on zero (e.g. MobileNetV3's hard-swish). 'performance' is
            symmetric and can be slightly faster on some devices.
        provenance: Extra provenance, e.g. the calibration dataset.
        overwrite: Replace an existing output bundle.
    """
    if len(calibration_images) == 0:
        raise ValueError("At least one calibration image is needed.")

    with open_bundle(path_source_bundle) as path_dir:
        model_info = ModelBundleInfo.from_json_file(path_dir / "model_info.json")
        runtime = model_info.runtime
        if runtime.model_format not in QUANTIZABLE_FORMATS or runtime.precision != "fp32":
            raise ValueError(
                f"Expected an FP32 bundle in one of the formats {QUANTIZABLE_FORMATS}, got model format "
                f"'{runtime.model_format}' with precision '{runtime.precision}'."
            )
        # Reads ONNX and OpenVINO IR. Quantization runs inside the 'with', while the extracted files exist.
        model = ov.Core().read_model(path_dir / runtime.file_path)

        calibration_dataset = nncf.Dataset(calibration_images, lambda image: preprocess_model_input(image, runtime)[0])
        ignored_patterns = runtime.quantization_hints.ignored_node_patterns if runtime.quantization_hints else []
        # 'validate=False': a pattern may match nothing after conversion, which is not an error
        ignored_scope = nncf.IgnoredScope(patterns=ignored_patterns, validate=False) if ignored_patterns else None
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf.QuantizationPreset(preset),
            subset_size=len(calibration_images),
            ignored_scope=ignored_scope,
        )

    quantization = {
        "tool": "nncf.quantize",
        "nncf_version": nncf.__version__,
        "preset": preset,
        "calibration_samples": len(calibration_images),
        "ignored_node_patterns": ignored_patterns,
    }
    return write_openvino_bundle(
        quantized_model,
        parent_info=model_info,
        path_bundle=path_bundle,
        precision="int8",
        provenance={"quantization": {**quantization, **(provenance or {})}},
        overwrite=overwrite,
    )
