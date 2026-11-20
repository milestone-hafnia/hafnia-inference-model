"""Standard NumPy components for pre- and postprocessing.

Importing this package registers them. Each is a pydantic model with its configuration as fields and its
implementation in 'forward', a thin wrapper around the functions in 'preprocessing', 'detection',
'classification' and 'primitives'.
"""

from inference_model_core.components.classification import ClassificationFromLogits
from inference_model_core.components.detection import AnchorNMS, Detections, DetectionsToBboxes, QueryTopK
from inference_model_core.components.image import ImageToTensor, LoadImage, ResizeImage
from inference_model_core.components.outputs import Predictions

__all__ = [
    "AnchorNMS",
    "ClassificationFromLogits",
    "Detections",
    "DetectionsToBboxes",
    "ImageToTensor",
    "LoadImage",
    "Predictions",
    "QueryTopK",
    "ResizeImage",
]
