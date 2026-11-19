"""ONNX export settings shared by the exporters."""

import warnings
from pathlib import Path

import torch
from torch import nn

DEFAULT_OPSET_VERSION = 17


def onnx_export(module: nn.Module, args: tuple, path_onnx: Path, **kwargs) -> None:
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
