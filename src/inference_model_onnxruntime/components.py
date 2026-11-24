"""ONNX Runtime: runs 'onnx_network' components in the torch graph engine."""

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import onnxruntime as ort

from inference_model_core.definitions import OnnxNetwork
from inference_model_torch.network import FileNetwork

ENGINE = "onnxruntime"

DEVICE_PROVIDERS: Dict[str, List[str]] = {
    "CPU": ["CPUExecutionProvider"],
    "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


class OnnxRuntimeNetwork(FileNetwork[OnnxNetwork]):
    engine = ENGINE

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

    def infer(self, arrays: Dict[str, np.ndarray]) -> Mapping[str, Any]:
        output_names = self.output_names
        return dict(zip(output_names, self._session.run(output_names, arrays), strict=True))
