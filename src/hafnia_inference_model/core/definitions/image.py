"""Definitions of image components: loading, resizing and conversion to a network input tensor."""

from typing import Any, List, Literal, Optional

from pydantic import Field, model_validator

from hafnia_inference_model.core.definition import ComponentDefinition, Port

ResizeMode = Literal["stretch", "letterbox", "keep_ratio_pad"]
Interpolation = Literal["linear", "nearest", "cubic", "area"]


class LoadImage(ComponentDefinition):
    """A file path, PIL image or array -> RGB image. Gray scale images are replicated to 3 channels and alpha
    channels are dropped."""

    type = "load_image"
    input_ports = (Port("image", "File path, PIL image or (H, W[, C]) uint8 array (RGB)"),)
    output_ports = (Port("image", "(H, W, 3) uint8 RGB image"),)


class ResizeImage(ComponentDefinition):
    """Resize an image to the network input size."""

    type = "resize_image"
    input_ports = (Port("image", "(H, W, 3) uint8 RGB image"),)
    output_ports = (
        Port("image", "(height, width, 3) uint8 RGB image"),
        Port("transform", "'ImageTransform' mapping network input coordinates back to the original image"),
    )

    mode: ResizeMode = Field(
        description=(
            "'stretch': resize to (height, width) ignoring aspect ratio. "
            "'letterbox': keep aspect ratio and pad equally on both sides. "
            "'keep_ratio_pad': keep aspect ratio and pad only bottom/right."
        )
    )
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    interpolation: Interpolation = "linear"
    antialias: bool = Field(
        default=False,
        description=(
            "Low-pass filter when downscaling, as torchvision's 'Resize' does by default on tensors (and PIL). "
            "Models trained with torchvision transforms need it to see the same pixels at inference"
        ),
    )
    pad_value: float = Field(default=0.0, description="Pixel value (0-255) used for padding")


class ImageToTensor(ComponentDefinition):
    """RGB image -> network input tensor without a batch axis.

    Steps are applied in this order: color_order -> scale -> normalize (mean/std) -> layout -> dtype."""

    type = "image_to_tensor"
    input_ports = (Port("image", "(H, W, 3) uint8 RGB image"),)
    output_ports = (Port("tensor", "(3, H, W) or (H, W, 3) tensor of 'dtype'"),)

    color_order: Literal["RGB", "BGR"] = "RGB"
    scale: Optional[float] = Field(default=1 / 255, description="Pixel multiplier applied before normalization")
    mean: Optional[List[float]] = Field(default=None, description="Per-channel mean, after scaling")
    std: Optional[List[float]] = Field(default=None, description="Per-channel std, after scaling")
    layout: Literal["CHW", "HWC"] = "CHW"
    dtype: Literal["float32", "float16", "uint8"] = "float32"

    @model_validator(mode="before")
    @classmethod
    def _no_default_scale_for_uint8(cls, data: Any) -> Any:
        # The default 'scale' (1/255) only applies to float tensors; 'uint8' tensors are never scaled
        if isinstance(data, dict) and data.get("dtype") == "uint8" and "scale" not in data:
            return {**data, "scale": None}
        return data

    @model_validator(mode="after")
    def _check_normalization(self) -> "ImageToTensor":
        if (self.mean is None) != (self.std is None):
            raise ValueError("'mean' and 'std' must either both be set or both be None.")
        if self.mean is not None and self.std is not None:
            if len(self.mean) != len(self.std):
                raise ValueError(f"'mean' ({len(self.mean)}) and 'std' ({len(self.std)}) must have the same length.")
            if any(s == 0 for s in self.std):
                raise ValueError("'std' must not contain zeros.")
        if self.dtype == "uint8" and (self.scale is not None or self.mean is not None):
            raise ValueError("'uint8' tensors can not be scaled or normalized. Set 'scale', 'mean' and 'std' to None.")
        return self
