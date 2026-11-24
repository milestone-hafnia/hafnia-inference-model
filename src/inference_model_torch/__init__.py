"""The torch graph engine: runs the model graph of a bundle with 'torch.Tensor's between the components.

Implements the standard component definitions in torch, runs torch networks stored by reference and converts
them to ONNX. Networks stored as ONNX or OpenVINO IR files run on their own inference engine inside the same
graph (see 'inference_model_onnxruntime' and 'inference_model_openvino')."""

from inference_model_torch.component import (
    Component,
    ComponentContext,
    LoadOptions,
    get_component_class,
    list_components,
    register_component,
)
from inference_model_torch.graph import TorchModelGraph

__all__ = [
    "Component",
    "ComponentContext",
    "LoadOptions",
    "TorchModelGraph",
    "get_component_class",
    "list_components",
    "register_component",
]
