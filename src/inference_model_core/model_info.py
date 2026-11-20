"""Model info: the machine-readable description stored as 'model_info.json' inside a model bundle.

'ModelBundleInfo' extends hafnia's 'ModelInfo' (name, tasks, references, ...) with:
- identity fields that survive conversions ('framework', 'variant')
- a 'graph': the ordered list of components that turn the graph inputs (e.g. an image) into the graph outputs
  (e.g. predictions). Each component reads and writes named items of one data dictionary.

A conversion (e.g. ONNX -> OpenVINO) copies the model info and only replaces the components it converts.

The extension lives here until the schema has settled and is then moved to hafnia (see docs/design.md).
"""

import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, Union

from hafnia.dataset.hafnia_dataset_types import ModelInfo
from pydantic import BaseModel, Field, field_validator, model_validator

MODEL_INFO_FORMAT_VERSION = "2.0.0"

Precision = Literal["fp32", "fp16", "int8"]

# Runtimes of this repo. Other packages can add their own, so the fields are plain strings.
RUNTIME_NUMPY = "numpy"
RUNTIME_TORCH = "torch"
RUNTIME_ONNX = "onnx"
RUNTIME_OPENVINO = "openvino"

_COMPONENT_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class TensorInfo(BaseModel):
    name: str
    shape: List[Optional[int]] = Field(description="Tensor shape. 'None' marks a dynamic dimension")
    dtype: str = "float32"


class NetworkParams(BaseModel):
    """Parameters of 'network' components: a tensor graph stored in model files (ONNX, OpenVINO IR).

    The component's input and output ports are the tensor names.
    """

    precision: Precision = "fp32"
    input_tensors: List[TensorInfo] = Field(min_length=1)
    output_tensors: List[TensorInfo] = Field(min_length=1)


class ConversionInfo(BaseModel):
    """Whether and how a component converts to one target (e.g. 'onnx', 'openvino', 'openvino_int8')."""

    supported: bool = True
    reason: Optional[str] = Field(default=None, description="Why the conversion is not supported")
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Export options of this component for the target, e.g. the ONNX opset and dynamic batch axis, or the "
            "graph nodes to keep in floating point when quantizing"
        ),
    )


class ComponentInfo(BaseModel):
    """One step of the model graph."""

    name: str = Field(description="Unique name in the graph, also the folder of the component's files")
    type: str = Field(description="What the component does, e.g. 'resize_image', 'network', 'anchor_nms'")
    runtime: str = Field(description="How the component is expressed: 'numpy', 'torch', 'onnx', 'openvino', ...")
    inputs: Dict[str, str] = Field(description="Input port (argument or tensor name) -> data dictionary key")
    outputs: Dict[str, str] = Field(description="Output port (result or tensor name) -> data dictionary key")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters of the component type")
    files: List[str] = Field(default_factory=list, description="Files of the component, relative to the bundle")
    batched: bool = Field(
        default=False,
        description=(
            "'True': called once per batch, with every input stacked on a new first axis. 'False': called once "
            "per sample"
        ),
    )
    conversions: Dict[str, ConversionInfo] = Field(
        default_factory=dict,
        description="Conversion target -> support and export options. A missing target can not be converted to",
    )
    provenance: Optional[Dict[str, Any]] = Field(
        default=None, description="E.g. the component it was converted from, tool versions, quantization settings"
    )

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str) -> str:
        if not _COMPONENT_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Component names are lower case identifiers ('a-z', '0-9', '_'), got '{name}'.")
        return name

    @field_validator("files")
    @classmethod
    def _check_files(cls, files: List[str]) -> List[str]:
        return [check_relative_bundle_path(path) for path in files]

    @model_validator(mode="after")
    def _check_ports(self) -> "ComponentInfo":
        if not self.outputs:
            raise ValueError(f"Component '{self.name}' has no outputs.")
        return self


class GraphInfo(BaseModel):
    """Components are run in list order. Every data key is written exactly once: by the caller (graph inputs)
    or by one component."""

    inputs: List[str] = Field(min_length=1, description="Data keys provided by the caller, e.g. ['image']")
    outputs: List[str] = Field(min_length=1, description="Data keys returned to the caller, e.g. ['predictions']")
    components: List[ComponentInfo] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_graph(self) -> "GraphInfo":
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError(f"Component names must be unique, got {names}.")
        all_files = self.all_files()
        if len(set(all_files)) != len(all_files):
            raise ValueError(f"Bundle files must be unique, got {all_files}.")

        available = set(self.inputs)
        if len(available) != len(self.inputs):
            raise ValueError(f"Graph inputs must be unique, got {self.inputs}.")
        for component in self.components:
            missing = sorted(set(component.inputs.values()) - available)
            if missing:
                raise ValueError(
                    f"Component '{component.name}' reads {missing}, which no graph input or earlier component "
                    f"provides. Available: {sorted(available)}."
                )
            written = list(component.outputs.values())
            rewritten = sorted({key for key in written if key in available or written.count(key) > 1})
            if rewritten:
                raise ValueError(f"Component '{component.name}' writes {rewritten}, which already exist.")
            available.update(written)
        missing_outputs = sorted(set(self.outputs) - available)
        if missing_outputs:
            raise ValueError(f"No component writes the graph outputs {missing_outputs}.")
        return self

    def all_files(self) -> List[str]:
        return [path for component in self.components for path in component.files]

    def get_component(self, name: str) -> ComponentInfo:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"No component '{name}' in the graph. Components: {[c.name for c in self.components]}")

    def components_for(self, outputs: List[str]) -> List[ComponentInfo]:
        """The components needed to compute the data keys 'outputs', in execution order."""
        produced = set(self.inputs) | {key for component in self.components for key in component.outputs.values()}
        unknown = sorted(set(outputs) - produced)
        if unknown:
            raise KeyError(f"The graph has no data keys {unknown}. Available: {sorted(produced)}")
        needed_keys = set(outputs)
        needed: List[ComponentInfo] = []
        for component in reversed(self.components):
            if needed_keys & set(component.outputs.values()):
                needed.append(component)
                needed_keys.update(component.inputs.values())
        return needed[::-1]


class ModelBundleInfo(ModelInfo):
    format_version: str = MODEL_INFO_FORMAT_VERSION
    framework: str = Field(description="Framework the model originates from, e.g. 'torchvision', 'rfdetr'")
    variant: Optional[str] = Field(
        default=None,
        description="Informational model variant/size, e.g. 'nano' or 'ssdlite320_mobilenet_v3_large'",
    )
    graph: GraphInfo

    @model_validator(mode="before")
    @classmethod
    def _check_format_version(cls, data: Any) -> Any:
        # Checked before the fields, so a bundle of another format version gets this error and not a list of
        # missing and unknown fields
        if isinstance(data, dict):
            format_version = str(data.get("format_version", MODEL_INFO_FORMAT_VERSION))
            supported_major = MODEL_INFO_FORMAT_VERSION.split(".")[0]
            if format_version.split(".")[0] != supported_major:
                raise ValueError(
                    f"Unsupported model info format version '{format_version}'. Supported major version is "
                    f"'{supported_major}': bundles of other versions have to be created again."
                )
        return data

    def all_files(self) -> List[str]:
        return self.graph.all_files()

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
