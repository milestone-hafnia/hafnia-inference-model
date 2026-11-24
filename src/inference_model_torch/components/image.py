"""Image components in torch: loading, resizing and conversion to a network input tensor."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hafnia.dataset.benchmark.inference_model import ImageType
from PIL import Image

from inference_model_core.definitions.image import ImageToTensor, Interpolation, LoadImage, ResizeImage, ResizeMode
from inference_model_core.transform import ImageTransform
from inference_model_torch.component import Component, register_component

_INTERPOLATION_MODES = {"linear": "bilinear", "nearest": "nearest", "cubic": "bicubic", "area": "area"}
_ANTIALIAS_MODES = ("bilinear", "bicubic")  # The modes torch can low-pass filter
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8}


@register_component
class TorchLoadImage(Component[LoadImage]):
    def forward(self, image: ImageType) -> Dict[str, Any]:
        return {"image": load_image_rgb(image).to(self.device)}


@register_component
class TorchResizeImage(Component[ResizeImage]):
    def setup(self) -> None:
        params: ResizeImage = self.params
        if params.antialias and _INTERPOLATION_MODES[params.interpolation] not in _ANTIALIAS_MODES:
            raise ValueError(
                f"Component '{self.info.name}': torch only supports 'antialias' for 'linear' and 'cubic' "
                f"interpolation, not '{params.interpolation}'."
            )

    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        params: ResizeImage = self.params
        resized, transform = resize_image(
            image,
            mode=params.mode,
            height=params.height,
            width=params.width,
            interpolation=params.interpolation,
            antialias=params.antialias,
            pad_value=params.pad_value,
        )
        return {"image": resized, "transform": transform}


@register_component
class TorchImageToTensor(Component[ImageToTensor]):
    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        params: ImageToTensor = self.params
        tensor = image_to_tensor(
            image,
            color_order=params.color_order,
            scale=params.scale,
            mean=params.mean,
            std=params.std,
            layout=params.layout,
            dtype=params.dtype,
        )
        return {"tensor": tensor}


def load_image_rgb(image: ImageType) -> torch.Tensor:
    """Return 'image' as an (H, W, 3) uint8 RGB tensor on the CPU.

    Accepts a file path, a PIL image, a numpy array or a tensor (assumed RGB, as returned by hafnia's
    'Sample.read_image'). Gray scale images are replicated to 3 channels and alpha channels are dropped.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as pil_image:
            return torch.from_numpy(np.array(pil_image.convert("RGB")))
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image.convert("RGB")))

    if isinstance(image, np.ndarray):
        # A tensor can't share the memory of a read-only array
        image = torch.from_numpy(image if image.flags.writeable else image.copy())
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported image type '{type(image).__name__}'.")
    if image.dtype != torch.uint8:
        raise ValueError(f"Expected a uint8 image, got dtype '{image.dtype}'.")
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Unsupported image shape {tuple(image.shape)}. Expected (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)."
        )
    if image.shape[2] == 1:
        return image.expand(-1, -1, 3)
    return image[:, :, :3]


def resize_image(
    image: torch.Tensor,
    mode: ResizeMode,
    height: int,
    width: int,
    interpolation: Interpolation = "linear",
    antialias: bool = False,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, ImageTransform]:
    """Resize an (H, W, 3) uint8 image to (height, width).

    Args:
        image: Image to resize.
        mode: 'stretch' ignores the aspect ratio. 'letterbox' keeps it and pads equally on both sides.
            'keep_ratio_pad' keeps it and pads only bottom/right.
        height: Target height.
        width: Target width.
        interpolation: Interpolation method.
        antialias: Low-pass filter when downscaling, as torchvision's 'Resize' does by default on tensors (and
            PIL). Only for 'linear' and 'cubic' interpolation.
        pad_value: Pixel value (0-255) used for padding.

    Returns:
        The resized (height, width, 3) uint8 image and how it maps to the original image.
    """
    interpolate_mode = _INTERPOLATION_MODES[interpolation]
    if antialias and interpolate_mode not in _ANTIALIAS_MODES:
        raise ValueError(
            f"'antialias' is only supported for 'linear' and 'cubic' interpolation, not '{interpolation}'."
        )
    original_height, original_width = image.shape[:2]
    pixels = image.permute(2, 0, 1)[None].to(torch.float32)  # (1, 3, H, W), as 'interpolate' takes it

    def resize_to(new_height: int, new_width: int) -> torch.Tensor:
        if interpolate_mode in _ANTIALIAS_MODES:
            return F.interpolate(
                pixels, size=(new_height, new_width), mode=interpolate_mode, align_corners=False, antialias=antialias
            )
        return F.interpolate(pixels, size=(new_height, new_width), mode=interpolate_mode)

    if mode == "stretch":
        resized = resize_to(height, width)
        transform = ImageTransform(
            original_height=original_height,
            original_width=original_width,
            input_height=height,
            input_width=width,
            scale_x=width / original_width,
            scale_y=height / original_height,
        )
        return _to_uint8_image(resized), transform

    # 'letterbox' and 'keep_ratio_pad' keep the aspect ratio and pad the remaining area
    scale = min(height / original_height, width / original_width)
    new_height = max(1, round(original_height * scale))
    new_width = max(1, round(original_width * scale))
    if mode == "letterbox":
        pad_top = (height - new_height) // 2
        pad_left = (width - new_width) // 2
    else:
        pad_top, pad_left = 0, 0

    canvas = pixels.new_full((1, 3, height, width), float(round(pad_value)))
    canvas[:, :, pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resize_to(new_height, new_width)
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
    return _to_uint8_image(canvas), transform


def _to_uint8_image(pixels: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) float pixels -> (H, W, 3) uint8, rounded as torchvision's 'resize' does for uint8 images."""
    return pixels[0].round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)


def image_to_tensor(
    image: torch.Tensor,
    color_order: str = "RGB",
    scale: Optional[float] = 1 / 255,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    layout: str = "CHW",
    dtype: str = "float32",
) -> torch.Tensor:
    """Turn an (H, W, 3) uint8 RGB image into a network input tensor without a batch axis.

    Steps, in this order: color order, pixel multiplier 'scale', per-channel '(x - mean) / std', layout, dtype.
    """
    if color_order == "BGR":
        image = image.flip(-1)
    tensor = image
    if scale is not None or mean is not None:
        tensor = image.to(torch.float32)
        if scale is not None:
            tensor = tensor * scale
        if mean is not None and std is not None:
            if len(mean) != tensor.shape[2]:
                raise ValueError(f"'mean'/'std' have {len(mean)} values, but the image has {tensor.shape[2]} channels.")
            tensor = (tensor - tensor.new_tensor(mean)) / tensor.new_tensor(std)
    if layout == "CHW":
        tensor = tensor.permute(2, 0, 1)
    return tensor.to(_DTYPES[dtype]).contiguous()
