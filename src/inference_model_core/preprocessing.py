"""Engine-agnostic image preprocessing functions, used by the image components ('components/image.py').

NumPy/OpenCV/PIL only: nothing here may depend on torch.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
from hafnia.dataset.benchmark.inference_model import ImageType
from PIL import Image

ResizeMode = Literal["stretch", "letterbox", "keep_ratio_pad"]
Interpolation = Literal["linear", "nearest", "cubic", "area"]

_INTERPOLATION_FLAGS = {
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}
# PIL filters antialias when downscaling, like torchvision's tensor 'Resize' (antialias=True): within 0.25 gray
# levels on average, where OpenCV's linear interpolation differs by 2-4 levels.
_PIL_FILTERS = {
    "linear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "cubic": Image.Resampling.BICUBIC,
    "area": Image.Resampling.BOX,
}


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

    def input_xyxy_to_normalized(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Map (N, 4) pixel boxes in model input space to normalized xyxy boxes on the original image,
        clipped to [0, 1]."""
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        x = (boxes[:, [0, 2]] - self.pad_left) / self.scale_x / self.original_width
        y = (boxes[:, [1, 3]] - self.pad_top) / self.scale_y / self.original_height
        normalized = np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], axis=1)
        return np.clip(normalized, 0.0, 1.0)


def load_image_rgb(image: ImageType) -> np.ndarray:
    """Return 'image' as an (H, W, 3) uint8 RGB array.

    Accepts a file path, a PIL image or a numpy array (assumed RGB, as returned by hafnia's
    'Sample.read_image'). Gray scale images are replicated to 3 channels and alpha channels are dropped.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return np.asarray(pil_image.convert("RGB"))

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type '{type(image).__name__}'.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, got dtype '{image.dtype}'.")
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported image shape {image.shape}. Expected (H, W), (H, W, 1), (H, W, 3) or (H, W, 4).")
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image[:, :, :3]


def resize_image(
    pixels: np.ndarray,
    mode: ResizeMode,
    height: int,
    width: int,
    interpolation: Interpolation = "linear",
    antialias: bool = False,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, ImageTransform]:
    """Resize an (H, W, 3) uint8 image to (height, width).

    Args:
        pixels: Image to resize.
        mode: 'stretch' ignores the aspect ratio. 'letterbox' keeps it and pads equally on both sides.
            'keep_ratio_pad' keeps it and pads only bottom/right.
        height: Target height.
        width: Target width.
        interpolation: Interpolation method.
        antialias: Low-pass filter when downscaling, as torchvision's 'Resize' does by default on tensors (and
            PIL). Models trained with torchvision transforms need it to see the same pixels at inference.
        pad_value: Pixel value (0-255) used for padding.

    Returns:
        The resized image and how it maps to the original image.
    """
    original_height, original_width = pixels.shape[:2]

    def resize_to(new_width: int, new_height: int) -> np.ndarray:
        if antialias:
            return np.asarray(Image.fromarray(pixels).resize((new_width, new_height), _PIL_FILTERS[interpolation]))
        return cv2.resize(pixels, (new_width, new_height), interpolation=_INTERPOLATION_FLAGS[interpolation])

    if mode == "stretch":
        transform = ImageTransform(
            original_height=original_height,
            original_width=original_width,
            input_height=height,
            input_width=width,
            scale_x=width / original_width,
            scale_y=height / original_height,
        )
        return resize_to(width, height), transform

    # 'letterbox' and 'keep_ratio_pad' keep the aspect ratio and pad the remaining area
    scale = min(height / original_height, width / original_width)
    new_height = max(1, round(original_height * scale))
    new_width = max(1, round(original_width * scale))
    resized = resize_to(new_width, new_height)

    if mode == "letterbox":
        pad_top = (height - new_height) // 2
        pad_left = (width - new_width) // 2
    else:
        pad_top, pad_left = 0, 0

    canvas = np.full((height, width, 3), round(pad_value), dtype=np.uint8)
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    transform = ImageTransform(
        original_height=original_height,
        original_width=original_width,
        input_height=height,
        input_width=width,
        scale_x=new_width / original_width,
        scale_y=new_height / original_height,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return canvas, transform


def image_to_tensor(
    pixels: np.ndarray,
    color_order: Literal["RGB", "BGR"] = "RGB",
    scale: Optional[float] = 1 / 255,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    layout: Literal["CHW", "HWC"] = "CHW",
    dtype: str = "float32",
) -> np.ndarray:
    """Turn an (H, W, 3) uint8 RGB image into a network input tensor without a batch axis.

    Steps, in this order: color order, pixel multiplier 'scale', per-channel '(x - mean) / std', layout, dtype.
    """
    if color_order == "BGR":
        pixels = pixels[:, :, ::-1]

    tensor = pixels.astype(np.float32)
    if scale is not None:
        tensor *= scale
    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=np.float32)
        if mean_array.shape[0] != tensor.shape[2]:
            raise ValueError(
                f"'mean'/'std' have {mean_array.shape[0]} values, but the image has {tensor.shape[2]} channels."
            )
        std_array = np.asarray(std, dtype=np.float32)
        tensor = (tensor - mean_array) / std_array
    if layout == "CHW":
        tensor = tensor.transpose(2, 0, 1)
    return np.ascontiguousarray(tensor.astype(dtype))
