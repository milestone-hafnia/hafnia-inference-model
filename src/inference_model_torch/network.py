"""Base classes of network implementations: networks stored as model files, and torch modules.

- 'FileNetwork': a network stored in model files (ONNX, OpenVINO IR) that an inference engine runs. The engine
  packages subclass it per engine. It takes and returns torch tensors, which it passes to the engine as NumPy
  arrays (without a copy on the CPU).
- 'TorchNetwork': a torch module stored by reference. The bundle holds the weights and the parameters of the
  definition name the code that builds the module (e.g. a torchvision architecture), so loading it needs the
  model's library. It is the source format of a model: 'export_onnx' gives the self-contained ONNX version
  ('inference_model_torch.convert').
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TypeVar

import numpy as np
import torch

from inference_model_core.definition import ComponentDefinition
from inference_model_core.definitions.network import NetworkDefinition
from inference_model_core.model_bundle import NetworkParams
from inference_model_torch.component import ENGINE, Component, from_numpy, to_numpy

N = TypeVar("N", bound=NetworkDefinition)
D = TypeVar("D", bound=ComponentDefinition)


class FileNetwork(Component[N]):
    """Subclasses set 'engine' and implement 'load_network' and 'infer'. The first file of the component is the
    model entry file (e.g. 'model.xml')."""

    def setup(self) -> None:
        context = self.context
        if not context.info.files or context.path_dir is None:
            raise ValueError(f"Network component '{context.info.name}' needs its model files to be loaded.")
        # Loaded from the file path so companion files next to the entry file are found ('model.bin', external data)
        self.load_network(context.path_dir / context.info.files[0])

    @abstractmethod
    def load_network(self, path_model: Path) -> None:
        """Load the model entry file with the engine, using 'self.context.options'."""

    @abstractmethod
    def infer(self, arrays: Dict[str, np.ndarray]) -> Mapping[str, Any]:
        """Input arrays by tensor name -> the connected outputs ('output_names') as arrays by tensor name."""

    def forward(self, **tensors: torch.Tensor) -> Dict[str, Any]:
        outputs = self.infer({name: to_numpy(tensor) for name, tensor in tensors.items()})
        return {name: from_numpy(array) for name, array in outputs.items()}

    @property
    def device(self) -> torch.device:
        """The engine gets its inputs from host memory, whatever device it computes on."""
        return torch.device("cpu")

    @property
    def output_names(self) -> List[str]:
        """The connected outputs: only these are computed."""
        return list(self.info.outputs)

    @property
    def max_batch_size(self) -> Optional[int]:
        """A network with a static batch axis takes batches of that size. 'None' for a dynamic batch axis."""
        if not self.info.batched:
            return None
        return self.params.input_tensors[0].shape[0]

    def check_graph_tensors(self, graph_inputs: set, graph_outputs: set, n_graph_inputs: int) -> None:
        """The model info must describe the stored graph: catch mismatches at load time instead of at inference.

        'graph_inputs' and 'graph_outputs' are all tensor names of the stored graph (a tensor can have several).
        """
        info_inputs = {tensor.name for tensor in self.params.input_tensors}
        info_outputs = {tensor.name for tensor in self.params.output_tensors}
        if not info_inputs <= graph_inputs or len(info_inputs) != n_graph_inputs or not info_outputs <= graph_outputs:
            raise ValueError(
                f"The model info of component '{self.info.name}' does not match its model file. Inputs: model info "
                f"{sorted(info_inputs)}, file {sorted(graph_inputs)}. Outputs: model info {sorted(info_outputs)}, "
                f"file {sorted(graph_outputs)}."
            )


@dataclass
class OnnxExport:
    """What 'export_onnx' reports about the ONNX file it wrote."""

    params: NetworkParams  # The tensors of the graph. Their names are the ports of the torch component
    batched: bool
    provenance: Dict[str, Any] = field(default_factory=dict)


class TorchNetwork(Component[D]):
    """Subclasses implement 'build_model', 'forward' (on 'self.model') and 'export_onnx'. The first file of the
    component is the weights file."""

    engine = ENGINE
    model: Any  # What 'build_model' returned

    def setup(self) -> None:
        context = self.context
        engine_options = context.options.for_engine(ENGINE)
        if engine_options:
            raise TypeError(f"Unknown torch engine options: {sorted(engine_options)}")
        if not context.info.files or context.path_dir is None:
            raise ValueError(f"Torch component '{context.info.name}' needs its weights file to be loaded.")
        self.model = self.build_model(context.path_dir / context.info.files[0], self.device)

    @abstractmethod
    def build_model(self, path_weights: Path, device: torch.device) -> Any:
        """Build the model from 'self.params' and load its weights. In eval mode, on 'device'."""

    @abstractmethod
    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        """Write the network as an ONNX file with the port names as tensor names.

        'options' are the component's stored 'onnx' conversion options, merged with the caller's.
        """


def load_state_dict_file(path_weights: Path) -> Dict[str, torch.Tensor]:
    """Read a state dict without unpickling arbitrary objects: weights of a bundle are data, never code."""
    return torch.load(path_weights, map_location="cpu", weights_only=True)
