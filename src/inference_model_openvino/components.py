"""OpenVINO: runs 'openvino_network' components (OpenVINO IR, 'model.xml' + 'model.bin') in the torch graph
engine, and 'onnx_network' components when a graph is loaded with 'inference_engine="openvino"'.

OpenVINO reads ONNX with its ONNX frontend, so the engine serves an ONNX network and its converted or quantized
IR versions.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, TypeVar

import numpy as np
import openvino as ov

from inference_model_core.definitions import NetworkDefinition, OnnxNetwork, OpenVINONetwork
from inference_model_torch import register_component
from inference_model_torch.network import FileNetwork

ENGINE = "openvino"
PERFORMANCE_HINTS = ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT")


N = TypeVar("N", bound=NetworkDefinition)


class _OpenVINONetwork(FileNetwork[N]):
    engine = ENGINE

    def load_network(self, path_model: Path) -> None:
        """Options: 'device' is an OpenVINO device name (e.g. 'CPU', 'GPU', 'NPU', 'AUTO'), 'num_threads' the
        inference threads. Engine option 'performance_hint': 'LATENCY', 'THROUGHPUT' or 'CUMULATIVE_THROUGHPUT'."""
        options = self.context.options
        engine_options = options.for_engine(ENGINE)
        performance_hint = engine_options.pop("performance_hint", None)
        if engine_options:
            raise TypeError(f"Unknown OpenVINO engine options: {sorted(engine_options)}")
        config: Dict[str, Any] = {}
        if options.num_threads is not None:
            config["INFERENCE_NUM_THREADS"] = options.num_threads
        if performance_hint is not None:
            if performance_hint.upper() not in PERFORMANCE_HINTS:
                raise ValueError(f"Unsupported performance hint '{performance_hint}'. Options: {PERFORMANCE_HINTS}")
            config["PERFORMANCE_HINT"] = performance_hint.upper()

        core = ov.Core()
        # Without mmap the weights are read into memory, so the extracted bundle can be deleted after loading
        core.set_property({"ENABLE_MMAP": False})
        compiled_model = core.compile_model(core.read_model(path_model), options.device, config)
        self.check_graph_tensors(
            _all_names(compiled_model.inputs),
            _all_names(compiled_model.outputs),
            n_graph_inputs=len(compiled_model.inputs),
        )
        # A graph tensor can have several names: the port of each connected output, by the model info's name
        self._output_ports: Dict[str, ov.Output] = {
            name: _port_named(compiled_model.outputs, name) for name in self.output_names
        }
        self._compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def infer(self, arrays: Dict[str, np.ndarray]) -> Mapping[str, Any]:
        # 'share_outputs=False' (default) copies the outputs, so they stay valid after the next inference
        results = self._infer_request.infer(arrays)
        return {name: results[port] for name, port in self._output_ports.items()}


@register_component
class OpenVINOIRNetwork(_OpenVINONetwork[OpenVINONetwork]):
    """The default implementation of networks stored as OpenVINO IR."""


@register_component(alternative=True)
class OpenVINOOnnxNetwork(_OpenVINONetwork[OnnxNetwork]):
    """ONNX networks on OpenVINO, for graphs loaded with 'inference_engine="openvino"'."""


def _all_names(ports: List[ov.Output]) -> Set[str]:
    return {name for port in ports for name in port.get_names()}


def _port_named(ports: List[ov.Output], name: str) -> ov.Output:
    return next(port for port in ports if name in port.get_names())
