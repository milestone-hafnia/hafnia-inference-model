"""The model bundle: a model stored as a 'model_info.json' and the files of its components.

'ModelBundle' is the description stored as 'model_info.json'. It extends hafnia's 'ModelInfo' (name, tasks,
references, ...) with:
- identity fields that survive conversions ('framework', 'variant')
- a 'graph': the ordered list of components that turn the graph inputs (e.g. an image) into the graph outputs
  (e.g. predictions). Each component reads and writes named items of one data dictionary.

A 'ModelBundle' also knows where its files are, once it is read from or written to a folder or a zip:

    bundle = ModelBundle.from_path("model_onnx.zip")  # Folder or zip. 'from_zip' only takes a zip
    bundle.write("model_onnx")  # As a folder, 'write_zip' as a zip
    bundle_openvino = bundle.convert("openvino", "model_openvino.zip")

The bundle only describes the model. What a component type means (its parameters and ports) is its component
definition ('inference_model_core.definition'), and how it is computed is up to the graph engine that runs the
bundle. A conversion (e.g. ONNX -> OpenVINO) copies the description and only replaces the components it converts.

The extension lives here until the schema has settled and is then moved to hafnia (see docs/design.md).
"""

import contextlib
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Mapping, Optional, Sequence

from hafnia.dataset.hafnia_dataset_types import ModelInfo
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from inference_model_core.bundle_io import (
    MODEL_INFO_FILENAME,
    FileSource,
    PathLike,
    check_relative_bundle_path,
    has_zip_suffix,
    is_zip_bundle,
    missing_bundle_files,
    open_bundle_dir,
    read_bundle_file,
    write_bundle_files,
)

MODEL_INFO_FORMAT_VERSION = "3.0.0"

Precision = Literal["fp32", "fp16", "int8"]

_COMPONENT_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


class TensorInfo(BaseModel):
    name: str
    shape: List[Optional[int]] = Field(description="Tensor shape. 'None' marks a dynamic dimension")
    dtype: str = "float32"


class NetworkParams(BaseModel):
    """Parameters of components that are a tensor graph stored in model files ('onnx_network',
    'openvino_network'). The component's input and output ports are the tensor names.
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
    type: str = Field(
        description=(
            "Component definition the component is an instance of, e.g. 'resize_image', 'anchor_nms', "
            "'onnx_network'. It names the parameters and ports, not how the component is computed"
        )
    )
    inputs: Dict[str, str] = Field(description="Input port (argument or tensor name) -> data dictionary key")
    outputs: Dict[str, str] = Field(description="Output port (result or tensor name) -> data dictionary key")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters of the component definition")
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


class ModelBundle(ModelInfo):
    format_version: str = MODEL_INFO_FORMAT_VERSION
    framework: str = Field(description="Framework the model originates from, e.g. 'torchvision', 'rfdetr'")
    variant: Optional[str] = Field(
        default=None,
        description="Informational model variant/size, e.g. 'nano' or 'ssdlite320_mobilenet_v3_large'",
    )
    graph: GraphInfo

    # Folder or zip holding the files. Not part of the description: it is not stored and not compared
    _path: Optional[Path] = PrivateAttr(default=None)

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

    def __eq__(self, other: Any) -> bool:
        # Bundles with the same description are equal, wherever their files are
        if not isinstance(other, ModelBundle):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    @property
    def path(self) -> Optional[Path]:
        """Folder or zip the bundle was read from or written to. 'None' for a bundle that is only in memory."""
        return self._path

    def all_files(self) -> List[str]:
        return self.graph.all_files()

    # --- reading -------------------------------------------------------------------------------------
    @classmethod
    def from_path(cls, path_bundle: PathLike) -> "ModelBundle":
        """Read a bundle from a folder or a zip.

        Only the model info is read. The model files are checked to be present, and stay where they are.
        """
        path_bundle = Path(path_bundle)
        bundle = cls.model_validate_json(read_bundle_file(path_bundle, MODEL_INFO_FILENAME))
        missing = missing_bundle_files(path_bundle, bundle.all_files())
        if missing:
            raise FileNotFoundError(f"Model bundle '{path_bundle}' is missing files of its model info: {missing}")
        bundle._path = path_bundle
        return bundle

    @classmethod
    def from_zip(cls, path_zip: PathLike) -> "ModelBundle":
        """Read a bundle from a zip. Like 'from_path', but fails on anything that is not a zip file."""
        if not is_zip_bundle(path_zip):
            raise ValueError(f"'{path_zip}' is not a zip file. Use 'from_path' for a bundle folder.")
        return cls.from_path(path_zip)

    def read_file(self, relative_path: str) -> bytes:
        """Read one file of the bundle, e.g. 'network/model.onnx', without extracting a zip bundle."""
        return read_bundle_file(self._source_path(), relative_path)

    @contextlib.contextmanager
    def open_dir(self) -> Iterator[Path]:
        """Yield a folder holding the files of the bundle.

        A folder bundle is yielded as-is. A zip bundle is extracted to a temporary folder that is deleted when
        the context exits, so load the model into memory inside the context.
        """
        with open_bundle_dir(self._source_path(), self.all_files()) as path_dir:
            yield path_dir

    # --- writing -------------------------------------------------------------------------------------
    def write(
        self,
        path_dir: PathLike,
        files: Optional[Mapping[str, FileSource]] = None,
        overwrite: bool = False,
    ) -> "ModelBundle":
        """Write the bundle as a folder.

        Args:
            path_dir: Folder to create.
            files: Bundle-relative path -> file source (a path or the content), for exactly the files the
                components list. By default the files of the bundle this one was read from.
            overwrite: Replace an existing folder or file at 'path_dir'.

        Returns:
            The written bundle: the same description, with 'path_dir' as its 'path'.
        """
        if has_zip_suffix(path_dir):
            raise ValueError(f"'write' creates a folder, but '{path_dir}' ends with '.zip'. Use 'write_zip'.")
        return self._write(path_dir, files, as_zip=False, overwrite=overwrite)

    def write_zip(
        self,
        path_zip: PathLike,
        files: Optional[Mapping[str, FileSource]] = None,
        overwrite: bool = False,
    ) -> "ModelBundle":
        """Write the bundle as a zip file. See 'write' for the arguments."""
        return self._write(path_zip, files, as_zip=True, overwrite=overwrite)

    def convert(
        self,
        target: str,
        path_bundle: PathLike,
        options: Optional[Mapping[str, Any]] = None,
        calibration_inputs: Optional[Sequence[Mapping[str, Any]]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        overwrite: bool = False,
    ) -> "ModelBundle":
        """Convert every component that supports 'target' and write the result as a new bundle.

        Args:
            target: Conversion target, e.g. 'onnx', 'openvino' or 'openvino_int8'.
            path_bundle: Output bundle. A path ending with '.zip' creates a zip, otherwise a folder.
            options: Options for the target that override the ones stored with the components, e.g.
                '{"precision": "fp16"}' for 'openvino'. They don't apply to earlier steps of a conversion chain.
            calibration_inputs: Graph inputs of representative samples (e.g. '[{"image": path}, ...]'), for targets
                that calibrate (quantization). Each converted component is calibrated on what it gets at inference:
                the samples are run through this bundle's graph up to that component.
            provenance: Extra provenance for the converted components, e.g. the calibration dataset.
            overwrite: Replace an existing output bundle.

        Returns:
            The converted bundle, with 'path_bundle' as its 'path'.
        """
        from inference_model_core.convert import convert_model_bundle  # Imported here: it imports this module

        return convert_model_bundle(
            self,
            path_bundle,
            target=target,
            options=options,
            calibration_inputs=calibration_inputs,
            provenance=provenance,
            overwrite=overwrite,
        )

    def _source_path(self) -> Path:
        if self._path is None:
            raise ValueError(
                f"Model bundle '{self.name}' is only in memory and has no files on disk. Write it with its 'files' "
                "first, or read a bundle with 'ModelBundle.from_path'."
            )
        return self._path

    def _write(
        self, path_bundle: PathLike, files: Optional[Mapping[str, FileSource]], as_zip: bool, overwrite: bool
    ) -> "ModelBundle":
        path_bundle = Path(path_bundle)
        if self._path is not None and path_bundle.resolve() == self._path.resolve():
            raise ValueError(f"Can't write model bundle '{path_bundle}' onto itself: its files are read from there.")
        model_info_json = self.model_dump_json(indent=4)
        if files is not None:
            expected_files = set(self.all_files())
            provided_files = {check_relative_bundle_path(path) for path in files}
            if provided_files != expected_files:
                # A bundle never holds files its model info doesn't describe
                raise ValueError(
                    "Files must match the files listed in the model info. "
                    f"Missing: {sorted(expected_files - provided_files)}, "
                    f"not listed: {sorted(provided_files - expected_files)}."
                )
            write_bundle_files(path_bundle, model_info_json, files, as_zip=as_zip, overwrite=overwrite)
        elif not self.all_files():
            write_bundle_files(path_bundle, model_info_json, {}, as_zip=as_zip, overwrite=overwrite)
        else:
            with self.open_dir() as path_source_dir:  # Raises for a bundle in memory: it has no files to copy
                sources: Dict[str, FileSource] = {path: path_source_dir / path for path in self.all_files()}
                write_bundle_files(path_bundle, model_info_json, sources, as_zip=as_zip, overwrite=overwrite)
        written = self.model_copy()
        written._path = path_bundle
        return written


def write_by_suffix(
    bundle: ModelBundle,
    path_bundle: PathLike,
    files: Optional[Mapping[str, FileSource]] = None,
    overwrite: bool = False,
) -> ModelBundle:
    """Write a bundle as a zip when 'path_bundle' ends with '.zip', otherwise as a folder.

    For functions that take one output path, e.g. 'ModelBundle.convert'.
    """
    writer = bundle.write_zip if has_zip_suffix(path_bundle) else bundle.write
    return writer(path_bundle, files=files, overwrite=overwrite)
