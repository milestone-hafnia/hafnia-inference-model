"""OpenVINO engine part shared by all OpenVINO flavors: loading, saving and raw inference.

OpenVINO runs both ONNX bundles (read with OpenVINO's ONNX frontend) and OpenVINO IR bundles
('model.xml' + 'model.bin'), so the same flavor serves an exported ONNX model and its converted
or quantized IR versions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import openvino as ov

from inference_model_core import InferenceModelBase, ModelBundleInfo, RawTensors, open_bundle, write_bundle
from inference_model_core.inference_model import PathLike

ENGINE = "openvino"
PERFORMANCE_HINTS = ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT")


class OpenVINOModel(InferenceModelBase):
    """Base class for OpenVINO flavors. Subclasses set 'framework', 'decoder', 'params_model' and
    implement 'postprocess'."""

    inference_engine = ENGINE
    supported_formats = ("onnx", "openvino")

    def __init__(
        self,
        model_info: ModelBundleInfo,
        compiled_model: ov.CompiledModel,
        model_files: Dict[str, bytes],
        postprocessing_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_info, postprocessing_params=postprocessing_params)
        self.compiled_model = compiled_model
        self._infer_request = compiled_model.create_infer_request()
        self._model_files = model_files  # Kept for 'write_model', as a compiled model can't be serialized back
        info_output_names = {tensor.name for tensor in model_info.runtime.outputs}
        self._output_names = [_tensor_name(output, info_output_names) for output in compiled_model.outputs]
        _check_tensor_names(model_info, compiled_model)

    @classmethod
    def load_model(
        cls,
        path_bundle: PathLike,
        postprocessing_params: Optional[Dict[str, Any]] = None,
        device: str = "CPU",
        num_threads: Optional[int] = None,
        performance_hint: Optional[str] = None,
        **engine_options: Any,
    ) -> "OpenVINOModel":
        """Load an ONNX or OpenVINO IR bundle.

        Args:
            path_bundle: Zip or directory bundle.
            postprocessing_params: Overrides of the decoder parameters.
            device: OpenVINO device name, e.g. 'CPU', 'GPU', 'NPU' or 'AUTO'.
            num_threads: Inference threads. 'None' uses the OpenVINO default.
            performance_hint: 'LATENCY', 'THROUGHPUT' or 'CUMULATIVE_THROUGHPUT'. 'None' uses the device default.
        """
        if engine_options:
            raise TypeError(f"Unknown OpenVINO engine options: {sorted(engine_options)}")
        config: Dict[str, Any] = {}
        if num_threads is not None:
            config["INFERENCE_NUM_THREADS"] = num_threads
        if performance_hint is not None:
            if performance_hint.upper() not in PERFORMANCE_HINTS:
                raise ValueError(f"Unsupported performance hint '{performance_hint}'. Options: {PERFORMANCE_HINTS}")
            config["PERFORMANCE_HINT"] = performance_hint.upper()

        core = ov.Core()
        # Without mmap the weights are read into memory, so the extracted bundle can be deleted after loading
        core.set_property({"ENABLE_MMAP": False})
        with open_bundle(path_bundle) as path_dir:
            model_info = ModelBundleInfo.from_json_file(path_dir / "model_info.json")
            model_files = {path: (path_dir / path).read_bytes() for path in model_info.runtime.all_files()}
            # Read from the file path so 'model.bin' (IR) and external data files (ONNX) are found
            model = core.read_model(path_dir / model_info.runtime.file_path)
            compiled_model = core.compile_model(model, device, config)
        return cls(model_info, compiled_model, model_files, postprocessing_params=postprocessing_params)

    def write_model(self, path_bundle: PathLike, overwrite: bool = False) -> Path:
        return write_bundle(path_bundle, self.model_info, self._model_files, overwrite=overwrite)

    def predict_raw(self, inputs: RawTensors) -> RawTensors:
        # 'share_outputs=False' (default) copies the outputs, so they stay valid after the next inference
        results = self._infer_request.infer(inputs)
        outputs = zip(self._output_names, self.compiled_model.outputs, strict=True)
        return {name: results[output] for name, output in outputs}


def _tensor_name(port: ov.Output, preferred_names: set) -> str:
    """A graph tensor can have several names. Use the one the model info refers to, if any."""
    names = port.get_names()
    matching = sorted(names & preferred_names)
    return matching[0] if matching else port.get_any_name()


def _check_tensor_names(model_info: ModelBundleInfo, compiled_model: ov.CompiledModel) -> None:
    """The model info must describe the graph: catch mismatches at load time instead of at inference."""

    def all_names(ports: List[ov.Output]) -> set:
        return {name for port in ports for name in port.get_names()}

    graph_inputs, graph_outputs = all_names(compiled_model.inputs), all_names(compiled_model.outputs)
    info_inputs = {tensor.name for tensor in model_info.runtime.input.tensors}
    info_outputs = {tensor.name for tensor in model_info.runtime.outputs}
    if (
        not info_inputs <= graph_inputs
        or len(info_inputs) != len(compiled_model.inputs)
        or (info_outputs and not info_outputs <= graph_outputs)
    ):
        raise ValueError(
            f"Model info does not match the OpenVINO model. Inputs: model info {sorted(info_inputs)}, "
            f"graph {sorted(graph_inputs)}. Outputs: model info {sorted(info_outputs)}, graph {sorted(graph_outputs)}."
        )
