"""Model info: the machine-readable description stored as 'model_info.json' inside a model bundle.

'ModelBundleInfo' extends hafnia's 'ModelInfo' (name, tasks, references, ...) with:
- identity fields that survive conversions ('framework', 'variant')
- a nested 'runtime' section describing how the model is stored and executed. An export (e.g.
  ONNX -> OpenVINO) copies the model info and only replaces the 'runtime' section.

The extension lives here until the schema has settled and is then moved to hafnia (see docs/design.md).
"""

from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Union

from hafnia.dataset.hafnia_dataset_types import ModelInfo
from pydantic import BaseModel, Field, field_validator, model_validator

MODEL_INFO_FORMAT_VERSION = "1.0.0"

ModelFormat = Literal["onnx", "openvino", "pt2", "tensorrt"]
InferenceEngine = Literal["onnxruntime", "openvino", "tensorrt", "torch"]
Precision = Literal["fp32", "fp16", "int8"]
PreprocessingStep = Literal["color_order", "resize", "scale", "normalize", "layout"]


class ResizeInfo(BaseModel):
    mode: Literal["stretch", "letterbox", "keep_ratio_pad", "none"] = Field(
        description=(
            "'stretch': resize to (height, width) ignoring aspect ratio. "
            "'letterbox': keep aspect ratio and pad equally on both sides. "
            "'keep_ratio_pad': keep aspect ratio and pad only bottom/right. "
            "'none': no resizing."
        )
    )
    height: Optional[int] = Field(default=None, description="Target height. 'None' takes it from the input shape")
    width: Optional[int] = Field(default=None, description="Target width. 'None' takes it from the input shape")
    interpolation: Literal["linear", "nearest", "cubic", "area"] = "linear"
    antialias: bool = Field(
        default=False,
        description=(
            "Low-pass filter when downscaling, as torchvision's 'Resize' does by default on tensors (and PIL). "
            "Models trained with torchvision transforms need it to see the same pixels at inference"
        ),
    )
    pad_value: float = Field(default=0.0, description="Pixel value (0-255) used for padding")


class PreprocessingInfo(BaseModel):
    """Declarative image -> input tensor steps, applied in this order:
    color_order -> resize -> scale -> normalize -> layout -> dtype.

    Steps listed in 'in_graph' are performed inside the model graph and are skipped outside of it."""

    color_order: Literal["RGB", "BGR"] = "RGB"
    resize: Optional[ResizeInfo] = None
    scale: Optional[float] = Field(default=1 / 255, description="Pixel multiplier applied before normalization")
    mean: Optional[List[float]] = Field(default=None, description="Per-channel mean, after scaling")
    std: Optional[List[float]] = Field(default=None, description="Per-channel std, after scaling")
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    dtype: Literal["float32", "float16", "uint8"] = "float32"
    in_graph: List[PreprocessingStep] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _no_default_scale_for_uint8(cls, data: Any) -> Any:
        # The default 'scale' (1/255) only applies to float inputs; 'uint8' inputs are never scaled
        if isinstance(data, dict) and data.get("dtype") == "uint8" and "scale" not in data:
            return {**data, "scale": None}
        return data

    @model_validator(mode="after")
    def _check_normalization(self) -> "PreprocessingInfo":
        if (self.mean is None) != (self.std is None):
            raise ValueError("'mean' and 'std' must either both be set or both be None.")
        if self.mean is not None and self.std is not None:
            if len(self.mean) != len(self.std):
                raise ValueError(f"'mean' ({len(self.mean)}) and 'std' ({len(self.std)}) must have the same length.")
            if any(s == 0 for s in self.std):
                raise ValueError("'std' must not contain zeros.")
        if self.dtype == "uint8" and (self.scale is not None or self.mean is not None):
            raise ValueError("'uint8' input can not be scaled or normalized. Set 'scale', 'mean' and 'std' to None.")
        return self

    def is_applied_outside_graph(self, step: PreprocessingStep) -> bool:
        return step not in self.in_graph


class PostprocessingInfo(BaseModel):
    decoder: str = Field(description="Name of the decoder (code) that turns raw outputs into primitives")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Decoder parameters, e.g. thresholds, output tensor names, box format, label mapping",
    )


class QuantizationHints(BaseModel):
    """Graph knowledge for post-training quantization, written by the exporter that built the graph.

    Quantizers stay model agnostic by reading these hints instead of knowing each model's graph.
    """

    ignored_node_patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Regular expressions (full match) of graph node names to keep in floating point, e.g. box "
            "decoding and score computation in the graph, where 8 bit values lose pixel and score precision"
        ),
    )
    model_type: Optional[Literal["transformer"]] = Field(
        default=None,
        description=(
            "Model architecture for the quantizer's defaults: 'transformer' keeps transformer specific operations "
            "(e.g. softmax, layer norm) in floating point. 'None' for convolutional networks"
        ),
    )


class TensorInfo(BaseModel):
    name: str
    shape: List[Optional[int]] = Field(description="Tensor shape. 'None' marks a dynamic dimension")
    dtype: str = "float32"


class InputInfo(BaseModel):
    modality: Literal["image", "video"] = Field(
        default="image", description="Only 'image' is supported in v1. 'video' is reserved for temporal models"
    )
    tensors: List[TensorInfo] = Field(min_length=1)


class RuntimeInfo(BaseModel):
    model_format: ModelFormat
    inference_engine: InferenceEngine = Field(
        description=(
            "Default engine for running the model. Callers can run the bundle on another engine that "
            "supports 'model_format', e.g. an ONNX bundle on OpenVINO"
        )
    )
    file_path: str = Field(description="Model entry file relative to the bundle root, e.g. 'model.xml'")
    extra_files: List[str] = Field(default_factory=list, description="Companion files, e.g. ['model.bin']")
    precision: Precision = "fp32"
    input: InputInfo
    outputs: List[TensorInfo] = Field(default_factory=list)
    preprocessing: PreprocessingInfo
    postprocessing: PostprocessingInfo
    quantization_hints: Optional[QuantizationHints] = None
    provenance: Optional[Dict[str, Any]] = Field(
        default=None, description="E.g. parent bundle, tool versions and quantization settings"
    )

    @field_validator("file_path")
    @classmethod
    def _check_file_path(cls, file_path: str) -> str:
        return check_relative_bundle_path(file_path)

    @field_validator("extra_files")
    @classmethod
    def _check_extra_files(cls, extra_files: List[str]) -> List[str]:
        return [check_relative_bundle_path(path) for path in extra_files]

    @model_validator(mode="after")
    def _check_unique_files(self) -> "RuntimeInfo":
        all_files = self.all_files()
        if len(set(all_files)) != len(all_files):
            raise ValueError(f"Bundle files must be unique, got {all_files}.")
        return self

    def all_files(self) -> List[str]:
        return [self.file_path, *self.extra_files]


class FlavorKey(NamedTuple):
    """Selects the 'InferenceModelBase' flavor that can load and run a model bundle."""

    inference_engine: str
    framework: str
    decoder: str


class ModelBundleInfo(ModelInfo):
    format_version: str = MODEL_INFO_FORMAT_VERSION
    framework: str = Field(description="Framework the model originates from, e.g. 'torchvision', 'rfdetr'")
    variant: Optional[str] = Field(
        default=None,
        description="Informational model variant/size, e.g. 'nano' or 'ssdlite320_mobilenet_v3_large'",
    )
    runtime: RuntimeInfo

    @field_validator("format_version")
    @classmethod
    def _check_format_version(cls, format_version: str) -> str:
        major = format_version.split(".")[0]
        supported_major = MODEL_INFO_FORMAT_VERSION.split(".")[0]
        if major != supported_major:
            raise ValueError(
                f"Unsupported model info format version '{format_version}'. "
                f"Supported major version is '{supported_major}'."
            )
        return format_version

    def flavor_key(self, inference_engine: Optional[str] = None) -> FlavorKey:
        """Flavor key for running this model, on 'inference_engine' or the default engine of the bundle."""
        return FlavorKey(
            inference_engine=inference_engine or self.runtime.inference_engine,
            framework=self.framework,
            decoder=self.runtime.postprocessing.decoder,
        )

    def write_json(self, path: Union[str, Path], indent: Optional[int] = 4) -> None:
        Path(path).write_text(self.model_dump_json(indent=indent))

    @staticmethod
    def from_json_file(path: Union[str, Path]) -> "ModelBundleInfo":
        return ModelBundleInfo.model_validate_json(Path(path).read_text())


def check_relative_bundle_path(path: str) -> str:
    """Bundle paths must be relative POSIX paths that stay inside the bundle root."""
    posix_path = PurePosixPath(path)
    if path == "" or "\\" in path or posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Bundle file paths must be relative paths inside the bundle, got '{path}'.")
    return posix_path.as_posix()
