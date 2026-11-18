"""ONNX Runtime engine part shared by all ONNX Runtime flavors: loading, saving and raw inference."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import onnxruntime as ort

from inference_model_core import InferenceModelBase, ModelBundleInfo, RawTensors, open_bundle, write_bundle
from inference_model_core.inference_model import PathLike

ENGINE = "onnxruntime"

DEVICE_PROVIDERS: Dict[str, List[str]] = {
    "CPU": ["CPUExecutionProvider"],
    "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


class OnnxRuntimeModel(InferenceModelBase):
    """Base class for ONNX Runtime flavors. Subclasses set 'framework', 'decoder', 'params_model' and
    implement 'postprocess'."""

    inference_engine = ENGINE
    supported_formats = ("onnx",)

    def __init__(
        self,
        model_info: ModelBundleInfo,
        session: ort.InferenceSession,
        model_files: Dict[str, bytes],
        postprocessing_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_info, postprocessing_params=postprocessing_params)
        self.session = session
        self._model_files = model_files  # Kept for 'write_model', as a session can't be serialized back
        self._output_names = [output.name for output in session.get_outputs()]
        _check_tensor_names(model_info, session)

    @classmethod
    def load_model(
        cls,
        path_bundle: PathLike,
        postprocessing_params: Optional[Dict[str, Any]] = None,
        device: str = "CPU",
        num_threads: Optional[int] = None,
        **engine_options: Any,
    ) -> "OnnxRuntimeModel":
        """Load an ONNX bundle.

        Args:
            path_bundle: Zip or directory bundle.
            postprocessing_params: Overrides of the decoder parameters.
            device: 'CPU' or 'CUDA'.
            num_threads: Intra-op threads. 'None' uses the ONNX Runtime default.
        """
        if engine_options:
            raise TypeError(f"Unknown ONNX Runtime engine options: {sorted(engine_options)}")
        if device.upper() not in DEVICE_PROVIDERS:
            raise ValueError(f"Unsupported device '{device}'. Options: {list(DEVICE_PROVIDERS)}")

        session_options = ort.SessionOptions()
        if num_threads is not None:
            session_options.intra_op_num_threads = num_threads

        with open_bundle(path_bundle) as path_dir:
            model_info = ModelBundleInfo.from_json_file(path_dir / "model_info.json")
            model_files = {path: (path_dir / path).read_bytes() for path in model_info.runtime.all_files()}
            # Created from the file path so external data files next to the model are found
            session = ort.InferenceSession(
                (path_dir / model_info.runtime.file_path).as_posix(),
                sess_options=session_options,
                providers=DEVICE_PROVIDERS[device.upper()],
            )
        return cls(model_info, session, model_files, postprocessing_params=postprocessing_params)

    def write_model(self, path_bundle: PathLike, overwrite: bool = False) -> Path:
        return write_bundle(path_bundle, self.model_info, self._model_files, overwrite=overwrite)

    def predict_raw(self, inputs: RawTensors) -> RawTensors:
        outputs = self.session.run(self._output_names, inputs)
        return dict(zip(self._output_names, outputs, strict=True))


def _check_tensor_names(model_info: ModelBundleInfo, session: ort.InferenceSession) -> None:
    """The model info must describe the graph: catch mismatches at load time instead of at inference."""
    graph_inputs = {tensor.name for tensor in session.get_inputs()}
    graph_outputs = {tensor.name for tensor in session.get_outputs()}
    info_inputs = {tensor.name for tensor in model_info.runtime.input.tensors}
    info_outputs = {tensor.name for tensor in model_info.runtime.outputs}
    if info_inputs != graph_inputs or (info_outputs and not info_outputs <= graph_outputs):
        raise ValueError(
            f"Model info does not match the ONNX graph. Inputs: model info {sorted(info_inputs)}, "
            f"graph {sorted(graph_inputs)}. Outputs: model info {sorted(info_outputs)}, graph {sorted(graph_outputs)}."
        )
