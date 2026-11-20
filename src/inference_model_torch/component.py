"""Base class of networks on the torch runtime.

A torch network is stored by reference: the bundle holds the weights and the component's fields name the code
that builds the module (e.g. a torchvision architecture). Loading it needs torch and the model's library. It is
the source format of a model: 'export_onnx' gives the self-contained ONNX version ('inference_model_torch.convert').
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from pydantic import PrivateAttr

from inference_model_core import Component, NetworkParams
from inference_model_core.bundle import FileSource
from inference_model_core.model_info import RUNTIME_TORCH

ENGINE = "torch"


@dataclass
class OnnxExport:
    """What 'export_onnx' reports about the ONNX file it wrote."""

    params: NetworkParams  # The tensors of the graph. Their names are the ports of the torch component
    batched: bool
    provenance: Dict[str, Any] = field(default_factory=dict)


class TorchNetworkComponent(Component):
    """Subclasses set 'type', declare what identifies the module as fields, and implement 'build_model',
    'forward' and 'export_onnx'. The first file of the component is the weights file.

    'forward' gets and returns NumPy arrays, as every component: 'call_model' runs a module on them.
    """

    runtime = RUNTIME_TORCH
    engine = ENGINE

    _model: Any = PrivateAttr()
    _device: torch.device = PrivateAttr()
    _weight_files: Dict[str, bytes] = PrivateAttr(default_factory=dict)  # Kept for writing the bundle again

    def setup(self) -> None:
        """Options: 'device' is 'CPU', 'CUDA' or a torch device name. 'num_threads' sets torch's global
        intra-op threads."""
        context = self.context
        engine_options = context.options.for_engine(ENGINE)
        if engine_options:
            raise TypeError(f"Unknown torch engine options: {sorted(engine_options)}")
        if not context.info.files or context.path_dir is None:
            raise ValueError(f"Torch component '{context.info.name}' needs its weights file to be loaded.")
        if context.options.num_threads is not None:
            torch.set_num_threads(context.options.num_threads)
        self._device = torch_device(context.options.device)
        self._model = self.build_model(context.path_dir / context.info.files[0], self._device)
        self._weight_files = {path: (context.path_dir / path).read_bytes() for path in context.info.files}

    def files(self) -> Dict[str, FileSource]:
        return dict(self._weight_files)

    def call_model(self, module: torch.nn.Module, *arrays: np.ndarray) -> Any:
        """Run a module on NumPy arrays: as tensors on the component's device, without gradients. Returns the
        module's outputs with every tensor as a NumPy array."""
        tensors = [torch.as_tensor(np.ascontiguousarray(array)).to(self._device) for array in arrays]
        with torch.no_grad():
            return _to_numpy(module(*tensors))

    @abstractmethod
    def build_model(self, path_weights: Path, device: torch.device) -> Any:
        """Build the model from the component's fields and load its weights. In eval mode, on 'device'."""

    @abstractmethod
    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        """Write the network as an ONNX file with the port names as tensor names.

        'options' are the component's stored 'onnx' conversion options, merged with the caller's.
        """


def _to_numpy(outputs: Any) -> Any:
    if isinstance(outputs, torch.Tensor):
        return outputs.detach().cpu().numpy()
    if isinstance(outputs, dict):
        return {name: _to_numpy(value) for name, value in outputs.items()}
    if isinstance(outputs, (tuple, list)):
        return tuple(_to_numpy(value) for value in outputs)
    return outputs


def torch_device(device: str) -> torch.device:
    names = {"CPU": "cpu", "CUDA": "cuda", "GPU": "cuda"}
    return torch.device(names.get(device.upper(), device))


def load_state_dict_file(path_weights: Path) -> Dict[str, torch.Tensor]:
    """Read a state dict without unpickling arbitrary objects: weights of a bundle are data, never code."""
    return torch.load(path_weights, map_location="cpu", weights_only=True)
