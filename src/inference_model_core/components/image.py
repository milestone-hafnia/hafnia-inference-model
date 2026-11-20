"""Image components: loading, resizing and conversion to a network input tensor."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference_model_core.component import NumpyComponent, register_component
from inference_model_core.preprocessing import (
    Interpolation,
    ResizeMode,
    image_to_tensor,
    load_image_rgb,
    resize_image,
)


@register_component
class LoadImage(NumpyComponent):
    """A file path, PIL image or array -> (H, W, 3) uint8 RGB array."""

    type = "load_image"
    input_ports = ("image",)
    output_ports = ("image",)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"image": load_image_rgb(inputs["image"])}


class ResizeImageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
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


@register_component
class ResizeImage(NumpyComponent):
    """Resize an image to the network input size. 'transform' maps network coordinates back to the original."""

    type = "resize_image"
    params_model = ResizeImageParams
    params: ResizeImageParams
    input_ports = ("image",)
    output_ports = ("image", "transform")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        params = self.params
        image, transform = resize_image(
            inputs["image"],
            mode=params.mode,
            height=params.height,
            width=params.width,
            interpolation=params.interpolation,
            antialias=params.antialias,
            pad_value=params.pad_value,
        )
        return {"image": image, "transform": transform}


class ImageToTensorParams(BaseModel):
    """Steps are applied in this order: color_order -> scale -> normalize (mean/std) -> layout -> dtype."""

    model_config = ConfigDict(extra="forbid")
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
    def _check_normalization(self) -> "ImageToTensorParams":
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


@register_component
class ImageToTensor(NumpyComponent):
    """(H, W, 3) uint8 RGB image -> network input tensor without a batch axis."""

    type = "image_to_tensor"
    params_model = ImageToTensorParams
    params: ImageToTensorParams
    input_ports = ("image",)
    output_ports = ("tensor",)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        params = self.params
        tensor = image_to_tensor(
            inputs["image"],
            color_order=params.color_order,
            scale=params.scale,
            mean=params.mean,
            std=params.std,
            layout=params.layout,
            dtype=params.dtype,
        )
        return {"tensor": tensor}
