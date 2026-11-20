"""ONNX export settings shared by the torch components."""

import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

import torch
from torch import nn

DEFAULT_OPSET_VERSION = 17


def onnx_export(module: nn.Module, args: tuple, path_onnx: Path, **kwargs: Any) -> None:
    """Export with the TorchScript-based exporter ('dynamo=False').

    It handles torchvision detection models (data dependent control flow in the transform and NMS). Tracer and
    user warnings about these constructs are expected; every exported graph is verified against torch in
    'tests/model_exporters'.
    """
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        torch.onnx.export(module.eval(), args, path_onnx.as_posix(), dynamo=False, do_constant_folding=True, **kwargs)


def onnx_options(options: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    """The component's ONNX export options: 'defaults' overridden by 'options'. Unknown options are an error."""
    unknown: Set[str] = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown ONNX export options {sorted(unknown)}. Options: {sorted(defaults)}")
    return {**defaults, **options}


def batch_axes(names: list, dynamic_batch: bool) -> Optional[Dict[str, Dict[int, str]]]:
    """'dynamic_axes' of 'torch.onnx.export' marking the first axis of the tensors 'names' as the batch."""
    return {name: {0: "batch"} for name in names} if dynamic_batch else None
