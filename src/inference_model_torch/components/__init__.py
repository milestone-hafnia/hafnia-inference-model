"""Torch implementations of the standard component definitions ('inference_model_core.definitions').

Importing this package registers them. The implementations are thin classes around functions on tensors, which
can also be used on their own.
"""

from inference_model_torch.components.classification import TorchClassificationFromLogits
from inference_model_torch.components.detection import TorchAnchorNMS, TorchDetectionsToBboxes, TorchQueryTopK
from inference_model_torch.components.image import TorchImageToTensor, TorchLoadImage, TorchResizeImage

__all__ = [
    "TorchAnchorNMS",
    "TorchClassificationFromLogits",
    "TorchDetectionsToBboxes",
    "TorchImageToTensor",
    "TorchLoadImage",
    "TorchQueryTopK",
    "TorchResizeImage",
]
