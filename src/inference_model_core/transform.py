"""'ImageTransform': the data item that describes how a resized image maps to the original image."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageTransform:
    """How an original image maps to the model input: 'input = original * scale + pad'."""

    original_height: int
    original_width: int
    input_height: int
    input_width: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_left: float = 0.0
    pad_top: float = 0.0

    @staticmethod
    def identity(height: int, width: int) -> "ImageTransform":
        """For boxes that already are in pixels of the original image."""
        return ImageTransform(original_height=height, original_width=width, input_height=height, input_width=width)
