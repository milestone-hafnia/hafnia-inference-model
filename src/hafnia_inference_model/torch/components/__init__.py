"""Torch implementations of the standard component definitions ('hafnia_inference_model.core.definitions').

Importing this package registers them. The implementations are thin classes around functions on tensors, which
can also be used on their own.
"""

from hafnia_inference_model.torch.components.classification import TorchClassificationFromLogits
from hafnia_inference_model.torch.components.detection import TorchAnchorNMS, TorchDetectionsToBboxes, TorchQueryTopK
from hafnia_inference_model.torch.components.image import TorchImageToTensor, TorchLoadImage, TorchResizeImage

__all__ = [
    "TorchAnchorNMS",
    "TorchClassificationFromLogits",
    "TorchDetectionsToBboxes",
    "TorchImageToTensor",
    "TorchLoadImage",
    "TorchQueryTopK",
    "TorchResizeImage",
]
