"""Converters of networks to 'openvino_network' components (OpenVINO IR, 'model.xml' + 'model.bin').

- 'openvino': an 'onnx_network' as IR, with FP32 or FP16 weights (option 'precision').
- 'openvino_int8': INT8 post-training quantization (NNCF) of an FP32 ONNX or IR network. The component is
  calibrated on the inputs it gets at inference ('ConversionRequest.calibration'). Which graph nodes stay in
  floating point and the model type (transformer or not) are the component's stored conversion options, written
  by the code that built the graph, so the converter stays model agnostic.
"""

from typing import Any, Dict, Iterator, Literal, Optional

import nncf
import openvino as ov

from inference_model_core import ComponentInfo, ConversionRequest, NetworkParams, register_converter
from inference_model_core.convert import TARGET_OPENVINO, TARGET_OPENVINO_INT8, CalibrationFeed
from inference_model_core.definitions.network import ONNX_NETWORK_TYPE, OPENVINO_NETWORK_TYPE
from inference_model_torch.component import to_numpy

IR_MODEL_FILENAME = "model.xml"
IR_WEIGHTS_FILENAME = "model.bin"
# Fewer calibration samples work, but NNCF advises at least 300 for reliable activation ranges
RECOMMENDED_CALIBRATION_SAMPLES = 300

QuantizationPreset = Literal["mixed", "performance"]

OPENVINO_OPTIONS = {"precision"}
INT8_OPTIONS = {"preset", "ignored_node_patterns", "model_type"}


@register_converter(ONNX_NETWORK_TYPE, TARGET_OPENVINO)
def convert_onnx_to_openvino(request: ConversionRequest) -> ComponentInfo:
    """Options:
    precision: Weight precision of the IR. 'fp32' (default) keeps the ONNX weights. 'fp16' stores weights as
        FP16, which halves the file size. Devices still choose their own compute precision.
    """
    _check_request(request, OPENVINO_OPTIONS)
    precision = request.options.get("precision", "fp32")
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"'precision' must be 'fp32' or 'fp16', got '{precision}'. INT8 is target 'openvino_int8'.")
    model = ov.convert_model(request.path_source_dir / request.component.files[0])
    provenance = {"conversion": {"tool": "openvino.convert_model", "source_type": request.component.type}}
    return _write_ir_component(request, model, precision, provenance)


@register_converter(ONNX_NETWORK_TYPE, TARGET_OPENVINO_INT8)
@register_converter(OPENVINO_NETWORK_TYPE, TARGET_OPENVINO_INT8)
def quantize_to_openvino_int8(request: ConversionRequest) -> ComponentInfo:
    """Quantize the weights and activations of an FP32 network to INT8.

    Options:
    preset: NNCF quantization preset. 'mixed' (default) quantizes activations asymmetrically, which suits
        activations that are not centered on zero (e.g. MobileNetV3's hard-swish). 'performance' is symmetric
        and can be slightly faster on some devices.
    ignored_node_patterns: Regular expressions (full match) of graph node names to keep in floating point, e.g.
        box decoding and score computation in the graph, where 8 bit values lose pixel and score precision.
    model_type: 'transformer' keeps transformer specific operations (e.g. softmax, layer norm) in floating
        point. 'None' for convolutional networks.
    """
    component = request.component
    _check_request(request, INT8_OPTIONS)
    if request.calibration is None:
        raise ValueError(f"Quantizing component '{component.name}' needs calibration inputs.")
    preset = request.options.get("preset", "mixed")
    ignored_patterns = list(request.options.get("ignored_node_patterns") or [])
    model_type = request.options.get("model_type")
    if model_type not in (None, "transformer"):
        raise ValueError(f"'model_type' must be 'transformer' or None, got '{model_type}'.")

    # Reads ONNX and OpenVINO IR
    model = ov.Core().read_model(request.path_source_dir / component.files[0])
    # 'validate=False': a pattern may match nothing after conversion, which is not an error
    ignored_scope = nncf.IgnoredScope(patterns=ignored_patterns, validate=False) if ignored_patterns else None
    quantized_model = nncf.quantize(
        model,
        nncf.Dataset(_Reiterable(request.calibration)),
        preset=nncf.QuantizationPreset(preset),
        subset_size=request.n_calibration_samples,
        ignored_scope=ignored_scope,
        model_type=nncf.ModelType.TRANSFORMER if model_type == "transformer" else None,
    )
    quantization = {
        "tool": "nncf.quantize",
        "nncf_version": nncf.__version__,
        "preset": preset,
        "calibration_samples": request.n_calibration_samples,
        "ignored_node_patterns": ignored_patterns,
        "model_type": model_type,
    }
    return _write_ir_component(request, quantized_model, "int8", {"quantization": quantization})


class _Reiterable:
    """NNCF can pass over the calibration data more than once. The feed yields the tensors of the torch graph
    engine, OpenVINO takes arrays."""

    def __init__(self, feed: CalibrationFeed):
        self._feed = feed

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for inputs in self._feed():
            yield {name: to_numpy(tensor) for name, tensor in inputs.items()}


def _check_request(request: ConversionRequest, known_options: set) -> NetworkParams:
    component = request.component
    if component.type not in (ONNX_NETWORK_TYPE, OPENVINO_NETWORK_TYPE):
        raise ValueError(
            f"Only '{ONNX_NETWORK_TYPE}' and '{OPENVINO_NETWORK_TYPE}' components convert to '{request.target}', "
            f"'{component.name}' is a '{component.type}'."
        )
    unknown = sorted(set(request.options) - known_options)
    if unknown:
        raise ValueError(f"Unknown options {unknown} for target '{request.target}'. Options: {sorted(known_options)}")
    params = NetworkParams.model_validate(component.params)
    if params.precision != "fp32":
        raise ValueError(
            f"Component '{component.name}' has precision '{params.precision}'. Only FP32 networks are converted."
        )
    return params


def _write_ir_component(
    request: ConversionRequest,
    model: ov.Model,
    precision: Literal["fp32", "fp16", "int8"],
    provenance: Optional[Dict[str, Any]] = None,
) -> ComponentInfo:
    """Write an OpenVINO model as the IR files of the converted component and return its component info.

    The source component (type, files, precision, provenance) is kept under 'provenance.parent', so a
    component can be traced back to its source. Weights are compressed to FP16 only for 'fp16': 'ov.save_model'
    compresses by default, which would silently turn an FP32 model into FP16.
    """
    component = request.component
    params = NetworkParams.model_validate(component.params)
    files = [f"{component.name}/{IR_MODEL_FILENAME}", f"{component.name}/{IR_WEIGHTS_FILENAME}"]
    (request.path_output_dir / component.name).mkdir(parents=True, exist_ok=True)
    ov.save_model(model, request.path_output_dir / files[0], compress_to_fp16=precision == "fp16")

    # Only an FP32 IR can still be quantized
    can_quantize = precision == "fp32" and TARGET_OPENVINO_INT8 in component.conversions
    conversions = {TARGET_OPENVINO_INT8: component.conversions[TARGET_OPENVINO_INT8]} if can_quantize else {}
    return component.model_copy(
        update={
            "type": OPENVINO_NETWORK_TYPE,
            "files": files,
            "params": params.model_copy(update={"precision": precision}).model_dump(mode="json"),
            "conversions": conversions,
            "provenance": {
                **(provenance or {}),
                "openvino_version": ov.get_version(),
                "compress_to_fp16": precision == "fp16",
                "parent": {
                    "type": component.type,
                    "files": component.files,
                    "precision": params.precision,
                    "provenance": component.provenance,
                },
            },
        }
    )
