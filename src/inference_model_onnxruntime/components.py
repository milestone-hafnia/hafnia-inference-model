"""ONNX Runtime: runs 'network' components stored as ONNX files."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from pydantic import PrivateAttr

from inference_model_core import register_component
from inference_model_core.model_info import RUNTIME_ONNX
from inference_model_core.network import NetworkComponent

ENGINE = "onnxruntime"

DEVICE_PROVIDERS: Dict[str, List[str]] = {
    "CPU": ["CPUExecutionProvider"],
    "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


@register_component
class OnnxRuntimeNetwork(NetworkComponent):
    runtime = RUNTIME_ONNX
    engine = ENGINE

    _session: ort.InferenceSession = PrivateAttr()

    def load_network(self, path_model: Path) -> None:
        """Options: 'device' is 'CPU' or 'CUDA', 'num_threads' the intra-op threads."""
        options = self.context.options
        engine_options = options.for_engine(ENGINE)
        if engine_options:
            raise TypeError(f"Unknown ONNX Runtime engine options: {sorted(engine_options)}")
        if options.device.upper() not in DEVICE_PROVIDERS:
            raise ValueError(f"Unsupported device '{options.device}'. Options: {list(DEVICE_PROVIDERS)}")

        session_options = ort.SessionOptions()
        if options.num_threads is not None:
            session_options.intra_op_num_threads = options.num_threads
        session = ort.InferenceSession(
            path_model.as_posix(), sess_options=session_options, providers=DEVICE_PROVIDERS[options.device.upper()]
        )
        graph_inputs = {tensor.name for tensor in session.get_inputs()}
        graph_outputs = {tensor.name for tensor in session.get_outputs()}
        self.check_graph_tensors(graph_inputs, graph_outputs, n_graph_inputs=len(graph_inputs))
        self._session = session

    def forward(self, **tensors: Any) -> Dict[str, Any]:
        output_names = self.output_names
        outputs = self._session.run(output_names, {name: np.asarray(value) for name, value in tensors.items()})
        return dict(zip(output_names, outputs, strict=True))
