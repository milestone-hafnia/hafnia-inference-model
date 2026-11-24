"""Definitions of networks stored as model files: a tensor graph that an inference library runs.

The ports of a network are its tensor names, which are part of its configuration. The first file of the
component is the model entry file ('model.onnx', 'model.xml').
"""

from inference_model_core.definition import ComponentDefinition, Ports
from inference_model_core.model_info import NetworkParams

ONNX_NETWORK_TYPE = "onnx_network"
OPENVINO_NETWORK_TYPE = "openvino_network"


class NetworkDefinition(NetworkParams, ComponentDefinition):
    def ports(self) -> Ports:
        return Ports(
            inputs=tuple(tensor.name for tensor in self.input_tensors),
            optional_inputs=(),
            outputs=tuple(tensor.name for tensor in self.output_tensors),
        )


class OnnxNetwork(NetworkDefinition):
    """A network stored as an ONNX file."""

    type = ONNX_NETWORK_TYPE


class OpenVINONetwork(NetworkDefinition):
    """A network stored as OpenVINO IR ('model.xml' + 'model.bin')."""

    type = OPENVINO_NETWORK_TYPE
