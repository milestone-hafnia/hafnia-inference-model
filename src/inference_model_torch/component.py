"""Base class of networks on the torch runtime.

A torch network is stored by reference: the bundle holds the weights and the component parameters name the code
that builds the module (e.g. a torchvision architecture). Loading it needs torch and the model's library. It is
the source format of a model: 'export_onnx' gives the self-contained ONNX version ('inference_model_torch.convert').
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import numpy as np
import torch
from pydantic import BaseModel

from inference_model_core import Component, ComponentInfo, LoadOptions, ModelBundleInfo, NetworkParams
from inference_model_core.bundle import FileSource
from inference_model_core.model_info import RUNTIME_TORCH

ENGINE = "torch"


@dataclass
class OnnxExport:
    """What 'export_onnx' reports about the ONNX file it wrote."""

    params: NetworkParams  # The tensors of the graph. Their names are the ports of the torch component
    batched: bool
    provenance: Dict[str, Any] = field(default_factory=dict)


T = TypeVar("T", bound="TorchNetworkComponent")


class TorchNetworkComponent(Component):
    """Subclasses set 'type', 'params_model' and the ports, and implement 'build_model', 'forward' and
    'export_onnx'. The first file of the component is the weights file."""

    runtime = RUNTIME_TORCH
    engine = ENGINE

    def __init__(
        self,
        info: ComponentInfo,
        model_info: ModelBundleInfo,
        model: Any,
        weight_files: Dict[str, bytes],
        device: torch.device,
    ):
        super().__init__(info, model_info)
        self.model = model
        self.device = device
        self._weight_files = weight_files  # Kept for writing the bundle again

    @classmethod
    def load(cls: Type[T], info: ComponentInfo, model_info: ModelBundleInfo, path_dir: Path, options: LoadOptions) -> T:
        """Options: 'device' is 'CPU', 'CUDA' or a torch device name. 'num_threads' sets torch's global
        intra-op threads."""
        engine_options = options.for_engine(ENGINE)
        if engine_options:
            raise TypeError(f"Unknown torch engine options: {sorted(engine_options)}")
        if not info.files:
            raise ValueError(f"Torch component '{info.name}' has no weights file.")
        if options.num_threads is not None:
            torch.set_num_threads(options.num_threads)
        device = torch_device(options.device)
        params = cls.params_model.model_validate(info.params)
        model = cls.build_model(params, path_dir / info.files[0], device)
        weight_files = {path: (path_dir / path).read_bytes() for path in info.files}
        return cls(info, model_info, model, weight_files, device)

    def files(self) -> Dict[str, FileSource]:
        return dict(self._weight_files)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tensors = {port: torch.as_tensor(np.ascontiguousarray(value)).to(self.device) for port, value in inputs.items()}
        with torch.no_grad():
            outputs = self.forward(tensors)
        return {port: tensor.detach().cpu().numpy() for port, tensor in outputs.items()}

    @classmethod
    @abstractmethod
    def build_model(cls, params: BaseModel, path_weights: Path, device: torch.device) -> Any:
        """Build the model from the component parameters and load its weights. In eval mode, on 'device'."""

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the model on tensors on its device (port -> tensor)."""

    @abstractmethod
    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        """Write the network as an ONNX file with the port names as tensor names.

        'options' are the component's stored 'onnx' conversion options, merged with the caller's.
        """


def torch_device(device: str) -> torch.device:
    names = {"CPU": "cpu", "CUDA": "cuda", "GPU": "cuda"}
    return torch.device(names.get(device.upper(), device))


def load_state_dict_file(path_weights: Path) -> Dict[str, torch.Tensor]:
    """Read a state dict without unpickling arbitrary objects: weights of a bundle are data, never code."""
    return torch.load(path_weights, map_location="cpu", weights_only=True)
