"""Torch implementations of the standard component definitions ('hafnia_model_bundle.core.definitions').

Importing this package registers them. The implementations are thin classes around functions on tensors, which
can also be used on their own.
"""

from hafnia_model_bundle.torch.components.classification import TorchClassificationFromLogits
from hafnia_model_bundle.torch.components.detection import TorchAnchorNMS, TorchDetectionsToBboxes, TorchQueryTopK
from hafnia_model_bundle.torch.components.image import TorchImageToTensor, TorchLoadImage, TorchResizeImage

__all__ = [
    "TorchAnchorNMS",
    "TorchClassificationFromLogits",
    "TorchDetectionsToBboxes",
    "TorchImageToTensor",
    "TorchLoadImage",
    "TorchQueryTopK",
    "TorchResizeImage",
]
