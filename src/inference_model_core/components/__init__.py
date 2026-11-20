"""Standard NumPy components for pre- and postprocessing.

Importing this package registers them. They are thin wrappers around the functions in 'preprocessing',
'detection', 'classification' and 'primitives', with typed parameters that are stored in the model info.
"""

from inference_model_core.components.classification import ClassificationFromLogits
from inference_model_core.components.detection import AnchorNMS, DetectionsToBboxes, QueryTopK
from inference_model_core.components.image import ImageToTensor, LoadImage, ResizeImage

__all__ = [
    "AnchorNMS",
    "ClassificationFromLogits",
    "DetectionsToBboxes",
    "ImageToTensor",
    "LoadImage",
    "QueryTopK",
    "ResizeImage",
]
