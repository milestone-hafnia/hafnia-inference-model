"""Base class of 'network' components: a tensor graph stored in model files and run by an inference engine.

The ports of a network are its tensor names. Engine packages subclass 'NetworkComponent' per (runtime, engine),
e.g. ONNX files on ONNX Runtime. The first file of the component is the model entry file (e.g. 'model.xml').
"""

from typing import Dict, Optional

from inference_model_core.bundle import FileSource
from inference_model_core.component import Component
from inference_model_core.model_info import ComponentInfo, ModelBundleInfo, NetworkParams

NETWORK_TYPE = "network"


class NetworkComponent(Component):
    type = NETWORK_TYPE
    params_model = NetworkParams
    params: NetworkParams

    def __init__(self, info: ComponentInfo, model_info: ModelBundleInfo, model_files: Dict[str, bytes]):
        super().__init__(info, model_info)
        if not info.files:
            raise ValueError(f"Network component '{info.name}' has no files.")
        # Kept for writing the bundle again, as engines can't serialize a loaded model back
        self._model_files = model_files
        self.output_names = list(info.outputs)

    def check_ports(self) -> None:
        input_tensors = {tensor.name for tensor in self.params.input_tensors}
        output_tensors = {tensor.name for tensor in self.params.output_tensors}
        if set(self.info.inputs) != input_tensors or not set(self.info.outputs) <= output_tensors:
            raise ValueError(
                f"The ports of network component '{self.info.name}' are its tensors: inputs "
                f"{sorted(input_tensors)} (all connected) and outputs {sorted(output_tensors)}. Got inputs "
                f"{sorted(self.info.inputs)} and outputs {sorted(self.info.outputs)}."
            )

    def files(self) -> Dict[str, FileSource]:
        return dict(self._model_files)

    @property
    def max_batch_size(self) -> Optional[int]:
        """A network with a static batch axis takes batches of that size. 'None' for a dynamic batch axis."""
        if not self.info.batched:
            return None
        return self.params.input_tensors[0].shape[0]

    def check_graph_tensors(self, graph_inputs: set, graph_outputs: set) -> None:
        """The model info must describe the stored graph: catch mismatches at load time instead of at inference.

        'graph_inputs' and 'graph_outputs' are all tensor names of the stored graph.
        """
        info_inputs = {tensor.name for tensor in self.params.input_tensors}
        info_outputs = {tensor.name for tensor in self.params.output_tensors}
        if not info_inputs <= graph_inputs or not info_outputs <= graph_outputs:
            raise ValueError(
                f"The model info of component '{self.info.name}' does not match its model file. Inputs: model info "
                f"{sorted(info_inputs)}, file {sorted(graph_inputs)}. Outputs: model info {sorted(info_outputs)}, "
                f"file {sorted(graph_outputs)}."
            )
