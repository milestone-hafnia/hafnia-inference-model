"""Converter of torch networks to 'onnx_network' components.

Generic over model families: the torch implementation of the family writes the ONNX file ('export_onnx'), using
its stored export options (e.g. opset, dynamic batch axis). It is registered once, as the entry point '*/onnx':
the converter of every component type that lists the target 'onnx' and has no converter of its own. A family
needs no entry for its network type.
"""

import onnx
import torch

from hafnia_inference_model.core import ComponentInfo, ConversionRequest
from hafnia_inference_model.core.convert import TARGET_ONNX
from hafnia_inference_model.core.definitions.network import ONNX_NETWORK_TYPE
from hafnia_inference_model.torch.component import get_component_class
from hafnia_inference_model.torch.network import TorchNetwork

ONNX_MODEL_FILENAME = "model.onnx"


def convert_torch_to_onnx(request: ConversionRequest) -> ComponentInfo:
    source = request.component
    component_class = get_component_class(source.type)
    if not issubclass(component_class, TorchNetwork):
        raise TypeError(
            f"Component '{source.name}' ('{source.type}') lists the conversion 'onnx' but is not a torch network: "
            "only a 'TorchNetwork' has an ONNX export."
        )
    component = component_class.load(source, request.bundle, request.path_source_dir)

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
        type=ONNX_NETWORK_TYPE,
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
            "parent": {"type": source.type, "params": source.params, "provenance": source.provenance},
        },
    )
