"""Generic, engine-agnostic image preprocessing driven by 'PreprocessingInfo'.

'preprocess_model_input' is the single implementation of model input preprocessing: it is the default
'InferenceModelBase.preprocess' and is used by exporters for quantization calibration, so calibration
always sees the same inputs as inference. NumPy/OpenCV only: nothing here may depend on torch.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from hafnia.dataset.benchmark.inference_model import ImageType
from PIL import Image

from inference_model_core.model_info import PreprocessingInfo, RuntimeInfo, TensorInfo

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
class PreprocessMeta:
    """How an original image maps to the model input: 'input = original * scale + pad'."""

    original_height: int
    original_width: int
    input_height: int
    input_width: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_left: float = 0.0
    pad_top: float = 0.0

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


def preprocess_image(
    image: ImageType,
    preprocessing: PreprocessingInfo,
    input_hw: Optional[Sequence[Optional[int]]] = None,
    add_batch_dim: bool = True,
) -> Tuple[np.ndarray, PreprocessMeta]:
    """Apply the preprocessing steps outside the graph to one image.

    Args:
        image: Input image (see 'load_image_rgb').
        preprocessing: Preprocessing description from the model info.
        input_hw: Spatial size of the model input, used when 'preprocessing.resize' has no explicit size.
        add_batch_dim: Prepend a batch axis of size 1.

    Returns:
        The model input tensor and the metadata to map outputs back to the original image.
    """
    image_rgb = load_image_rgb(image)
    original_height, original_width = image_rgb.shape[:2]

    pixels = image_rgb
    if preprocessing.color_order == "BGR" and preprocessing.is_applied_outside_graph("color_order"):
        pixels = pixels[:, :, ::-1]

    meta = PreprocessMeta(
        original_height=original_height,
        original_width=original_width,
        input_height=original_height,
        input_width=original_width,
    )
    resize = preprocessing.resize
    if resize is not None and resize.mode != "none" and preprocessing.is_applied_outside_graph("resize"):
        pixels, meta = _resize(pixels, preprocessing, input_hw)

    tensor = pixels.astype(np.float32)
    if preprocessing.scale is not None and preprocessing.is_applied_outside_graph("scale"):
        tensor *= preprocessing.scale

    if preprocessing.mean is not None and preprocessing.is_applied_outside_graph("normalize"):
        mean = np.asarray(preprocessing.mean, dtype=np.float32)
        std = np.asarray(preprocessing.std, dtype=np.float32)
        if mean.shape[0] != tensor.shape[2]:
            raise ValueError(f"'mean'/'std' have {mean.shape[0]} values, but the image has {tensor.shape[2]} channels.")
        tensor = (tensor - mean) / std

    if preprocessing.layout == "NCHW" and preprocessing.is_applied_outside_graph("layout"):
        tensor = tensor.transpose(2, 0, 1)

    tensor = np.ascontiguousarray(tensor.astype(preprocessing.dtype))
    if add_batch_dim:
        tensor = tensor[None]
    return tensor, meta


def _resize(
    pixels: np.ndarray,
    preprocessing: PreprocessingInfo,
    input_hw: Optional[Sequence[Optional[int]]],
) -> Tuple[np.ndarray, PreprocessMeta]:
    resize = preprocessing.resize
    assert resize is not None
    input_height, input_width = input_hw if input_hw is not None else (None, None)
    target_height = resize.height or input_height
    target_width = resize.width or input_width
    if target_height is None or target_width is None:
        raise ValueError(
            "Resize target size is unknown. Set 'resize.height'/'resize.width' or use a model input with a static "
            "spatial size."
        )

    original_height, original_width = pixels.shape[:2]

    def resize_to(width: int, height: int) -> np.ndarray:
        if resize.antialias:
            return np.asarray(Image.fromarray(pixels).resize((width, height), _PIL_FILTERS[resize.interpolation]))
        return cv2.resize(pixels, (width, height), interpolation=_INTERPOLATION_FLAGS[resize.interpolation])

    if resize.mode == "stretch":
        resized = resize_to(target_width, target_height)
        meta = PreprocessMeta(
            original_height=original_height,
            original_width=original_width,
            input_height=target_height,
            input_width=target_width,
            scale_x=target_width / original_width,
            scale_y=target_height / original_height,
        )
        return resized, meta

    # 'letterbox' and 'keep_ratio_pad' keep the aspect ratio and pad the remaining area
    scale = min(target_height / original_height, target_width / original_width)
    new_height = max(1, round(original_height * scale))
    new_width = max(1, round(original_width * scale))
    resized = resize_to(new_width, new_height)

    if resize.mode == "letterbox":
        pad_top = (target_height - new_height) // 2
        pad_left = (target_width - new_width) // 2
    else:
        pad_top, pad_left = 0, 0

    canvas = np.full((target_height, target_width, 3), round(resize.pad_value), dtype=np.uint8)
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    meta = PreprocessMeta(
        original_height=original_height,
        original_width=original_width,
        input_height=target_height,
        input_width=target_width,
        scale_x=new_width / original_width,
        scale_y=new_height / original_height,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return canvas, meta


def input_spatial_size(tensor: TensorInfo, preprocessing: PreprocessingInfo) -> Tuple[Optional[int], Optional[int]]:
    """(height, width) of an image input tensor of rank 3 (no batch axis) or rank 4. 'None' if dynamic."""
    if len(tensor.shape) not in (3, 4):
        raise ValueError(f"Image input '{tensor.name}' must have rank 3 or 4, got shape {tensor.shape}.")
    channels_first = preprocessing.layout == "NCHW" and preprocessing.is_applied_outside_graph("layout")
    spatial = tensor.shape[-2:] if channels_first else tensor.shape[-3:-1]
    return spatial[0], spatial[1]


def preprocess_model_input(image: ImageType, runtime: RuntimeInfo) -> Tuple[Dict[str, np.ndarray], PreprocessMeta]:
    """Turn one image into the inputs of a single-image model call, as described by the model info.

    The image goes to the first input tensor. A rank 4 input gets a batch axis of size 1, a rank 3 input
    (e.g. torchvision detection models exported with their transform) gets none.
    """
    tensor_info = runtime.input.tensors[0]
    tensor, meta = preprocess_image(
        image,
        runtime.preprocessing,
        input_hw=input_spatial_size(tensor_info, runtime.preprocessing),
        add_batch_dim=len(tensor_info.shape) == 4,
    )
    return {tensor_info.name: tensor}, meta
