"""ONNX Runtime: runs 'network' components stored as ONNX files."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort

from inference_model_core import ComponentInfo, LoadOptions, ModelBundleInfo, register_component
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

    def __init__(
        self,
        info: ComponentInfo,
        model_info: ModelBundleInfo,
        session: ort.InferenceSession,
        model_files: Dict[str, bytes],
    ):
        super().__init__(info, model_info, model_files)
        self.session = session
        graph_inputs = {tensor.name for tensor in session.get_inputs()}
        if len(graph_inputs) != len(self.params.input_tensors):
            raise ValueError(f"Component '{info.name}' must list all inputs of its ONNX graph: {sorted(graph_inputs)}.")
        self.check_graph_tensors(graph_inputs, {tensor.name for tensor in session.get_outputs()})

    @classmethod
    def load(
        cls, info: ComponentInfo, model_info: ModelBundleInfo, path_dir: Path, options: LoadOptions
    ) -> "OnnxRuntimeNetwork":
        """Options: 'device' is 'CPU' or 'CUDA', 'num_threads' the intra-op threads."""
        engine_options = options.for_engine(ENGINE)
        if engine_options:
            raise TypeError(f"Unknown ONNX Runtime engine options: {sorted(engine_options)}")
        if options.device.upper() not in DEVICE_PROVIDERS:
            raise ValueError(f"Unsupported device '{options.device}'. Options: {list(DEVICE_PROVIDERS)}")

        session_options = ort.SessionOptions()
        if options.num_threads is not None:
            session_options.intra_op_num_threads = options.num_threads
        model_files = {path: (path_dir / path).read_bytes() for path in info.files}
        # Created from the file path so external data files next to the model are found
        session = ort.InferenceSession(
            (path_dir / info.files[0]).as_posix(),
            sess_options=session_options,
            providers=DEVICE_PROVIDERS[options.device.upper()],
        )
        return cls(info, model_info, session, model_files)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self.session.run(self.output_names, {name: np.asarray(value) for name, value in inputs.items()})
        return dict(zip(self.output_names, outputs, strict=True))
