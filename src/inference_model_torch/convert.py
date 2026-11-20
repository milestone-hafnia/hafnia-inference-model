"""Converter of torch networks to ONNX 'network' components.

Generic over model families: the torch component of the family writes the ONNX file ('export_onnx'), using its
stored export options (e.g. opset, dynamic batch axis).
"""

import onnx
import torch

from inference_model_core import ComponentInfo, ConversionRequest, LoadOptions, get_component_class, register_converter
from inference_model_core.convert import TARGET_ONNX
from inference_model_core.model_info import RUNTIME_ONNX, RUNTIME_TORCH
from inference_model_core.network import NETWORK_TYPE
from inference_model_torch.component import TorchNetworkComponent

ONNX_MODEL_FILENAME = "model.onnx"


@register_converter(RUNTIME_TORCH, TARGET_ONNX)
def convert_torch_to_onnx(request: ConversionRequest) -> ComponentInfo:
    source = request.component
    component_class = get_component_class(source.type, RUNTIME_TORCH)
    if not issubclass(component_class, TorchNetworkComponent):
        raise TypeError(f"Component '{source.name}' ('{source.type}') is not a torch network and has no ONNX export.")
    component = component_class.load(source, request.model_info, request.path_source_dir, LoadOptions())

    path_file = f"{source.name}/{ONNX_MODEL_FILENAME}"
    path_onnx = request.path_output_dir / path_file
    path_onnx.parent.mkdir(parents=True, exist_ok=True)
    export = component.export_onnx(path_onnx, request.options)
    onnx.checker.check_model(path_onnx.as_posix())

    tensor_names = {tensor.name for tensor in [*export.params.input_tensors, *export.params.output_tensors]}
    ports = {*source.inputs, *source.outputs}
    if not ports <= tensor_names:
        raise ValueError(
            f"The ONNX export of component '{source.name}' has tensors {sorted(tensor_names)}, which don't cover "
            f"the component's ports {sorted(ports)}."
        )
    return ComponentInfo(
        name=source.name,
        type=NETWORK_TYPE,
        runtime=RUNTIME_ONNX,
        inputs=source.inputs,
        outputs=source.outputs,
        params=export.params.model_dump(mode="json"),
        files=[path_file],
        batched=export.batched,
        # The remaining conversions (e.g. to OpenVINO) apply to the ONNX network
        conversions={name: info for name, info in source.conversions.items() if name != TARGET_ONNX},
        provenance={
            **export.provenance,
            "conversion": {"tool": "torch.onnx.export", "torch": torch.__version__, "options": request.options},
            "parent": {
                "type": source.type,
                "runtime": source.runtime,
                "params": source.params,
                "provenance": source.provenance,
            },
        },
    )
