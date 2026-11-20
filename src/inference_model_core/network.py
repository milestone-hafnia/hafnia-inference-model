"""Base class of 'network' components: a tensor graph stored in model files and run by an inference engine.

The ports of a network are its tensor names, which are part of its configuration, so 'forward' takes and returns
the tensors by name. Engine packages subclass 'NetworkComponent' per (runtime, engine), e.g. ONNX files on ONNX
Runtime. The first file of the component is the model entry file (e.g. 'model.xml').
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from inference_model_core.bundle import FileSource
from inference_model_core.component import Component, Ports
from inference_model_core.model_info import NetworkParams

NETWORK_TYPE = "network"


class NetworkComponent(NetworkParams, Component):
    """Subclasses implement 'load_network' and 'forward'."""

    type = NETWORK_TYPE

    # Kept for writing the bundle again, as engines can't serialize a loaded model back
    _model_files: Dict[str, bytes] = PrivateAttr(default_factory=dict)

    def ports(self) -> Ports:
        return Ports(
            inputs=tuple(tensor.name for tensor in self.input_tensors),
            optional_inputs=(),
            outputs=tuple(tensor.name for tensor in self.output_tensors),
        )

    @abstractmethod
    def forward(self, **tensors: Any) -> Dict[str, Any]:
        """Input tensors by name -> the connected output tensors by name."""

    def setup(self) -> None:
        context = self.context
        if not context.info.files or context.path_dir is None:
            raise ValueError(f"Network component '{context.info.name}' needs its model files to be loaded.")
        self._model_files = {path: (context.path_dir / path).read_bytes() for path in context.info.files}
        # Loaded from the file path so companion files next to the entry file are found ('model.bin', external data)
        self.load_network(context.path_dir / context.info.files[0])

    @abstractmethod
    def load_network(self, path_model: Path) -> None:
        """Load the model entry file with the engine, using 'self.context.options'."""

    def files(self) -> Dict[str, FileSource]:
        return dict(self._model_files)

    @property
    def output_names(self) -> List[str]:
        """The connected outputs: only these are computed."""
        return list(self.info.outputs)

    @property
    def max_batch_size(self) -> Optional[int]:
        """A network with a static batch axis takes batches of that size. 'None' for a dynamic batch axis."""
        if not self.info.batched:
            return None
        return self.input_tensors[0].shape[0]

    def check_graph_tensors(self, graph_inputs: set, graph_outputs: set, n_graph_inputs: int) -> None:
        """The model info must describe the stored graph: catch mismatches at load time instead of at inference.

        'graph_inputs' and 'graph_outputs' are all tensor names of the stored graph (a tensor can have several).
        """
        info_inputs = {tensor.name for tensor in self.input_tensors}
        info_outputs = {tensor.name for tensor in self.output_tensors}
        if not info_inputs <= graph_inputs or len(info_inputs) != n_graph_inputs or not info_outputs <= graph_outputs:
            raise ValueError(
                f"The model info of component '{self.info.name}' does not match its model file. Inputs: model info "
                f"{sorted(info_inputs)}, file {sorted(graph_inputs)}. Outputs: model info {sorted(info_outputs)}, "
                f"file {sorted(graph_outputs)}."
            )
