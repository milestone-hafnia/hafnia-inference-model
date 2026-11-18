"""Core of the generic inference model concept: model info, model bundles, the 'InferenceModelBase' class
and the flavor registry. Engine agnostic and torch free (see docs/design.md)."""

from inference_model_core.bundle import (
    MODEL_INFO_FILENAME,
    open_bundle,
    read_bundle_file,
    read_model_info,
    validate_bundle,
    write_bundle,
)
from inference_model_core.inference_model import InferenceModelBase, RawTensors
from inference_model_core.model_info import (
    FlavorKey,
    InputInfo,
    ModelBundleInfo,
    PostprocessingInfo,
    PreprocessingInfo,
    ResizeInfo,
    RuntimeInfo,
    TensorInfo,
)
from inference_model_core.preprocessing import PreprocessMeta, load_image_rgb, preprocess_image
from inference_model_core.registry import get_flavor, inference_model_from_path, list_flavors, register_flavor

__all__ = [
    "MODEL_INFO_FILENAME",
    "FlavorKey",
    "InferenceModelBase",
    "InputInfo",
    "ModelBundleInfo",
    "PostprocessingInfo",
    "PreprocessMeta",
    "PreprocessingInfo",
    "RawTensors",
    "ResizeInfo",
    "RuntimeInfo",
    "TensorInfo",
    "get_flavor",
    "inference_model_from_path",
    "list_flavors",
    "load_image_rgb",
    "open_bundle",
    "preprocess_image",
    "read_bundle_file",
    "read_model_info",
    "register_flavor",
    "validate_bundle",
    "write_bundle",
]
