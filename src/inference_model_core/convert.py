"""Generic conversion of model bundles: 'convert_bundle' and the registry of converters.

A bundle is converted component by component. Each component lists in its 'conversions' the targets it supports
(e.g. 'onnx', 'openvino', 'openvino_int8') with its export options. A converter turns a component of one runtime
into a target; chains such as torch -> onnx -> openvino are found automatically. Components that don't list the
target are copied unchanged, so e.g. NumPy processing stays as it is next to a converted network.

Converters register with 'register_converter'. Separately installed packages list theirs in the
'hafnia.inference_model_converters' entry point group, named '<source runtime>/<target>': they are imported only
when used.
"""

import tempfile
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from hafnia.log import sys_logger

from inference_model_core.bundle import MODEL_INFO_FILENAME, FileSource, PathLike, open_bundle, write_bundle
from inference_model_core.graph import ModelGraph
from inference_model_core.model_info import (
    RUNTIME_ONNX,
    RUNTIME_OPENVINO,
    ComponentInfo,
    GraphInfo,
    ModelBundleInfo,
)

ENTRY_POINT_GROUP = "hafnia.inference_model_converters"

TARGET_ONNX = "onnx"
TARGET_OPENVINO = "openvino"
TARGET_OPENVINO_INT8 = "openvino_int8"
# Runtime of a component after it was converted to a target
TARGET_RUNTIMES: Dict[str, str] = {
    TARGET_ONNX: RUNTIME_ONNX,
    TARGET_OPENVINO: RUNTIME_OPENVINO,
    TARGET_OPENVINO_INT8: RUNTIME_OPENVINO,
}

# Yields the inputs of the component being converted (port -> array), as it gets them at inference
CalibrationFeed = Callable[[], Iterator[Dict[str, Any]]]


@dataclass(frozen=True)
class ConversionRequest:
    """What a converter gets.

    The converter writes the files of the converted component below 'path_output_dir' (in the component's
    folder) and returns its component info, with the same name and outputs as the source component.
    """

    component: ComponentInfo
    model_info: ModelBundleInfo
    path_source_dir: Path  # Directory holding the files of 'component', at their bundle paths
    path_output_dir: Path
    target: str
    options: Dict[str, Any]  # The component's stored options for the target, merged with the caller's
    calibration: Optional[CalibrationFeed]  # 'None' when the caller gave no calibration inputs
    n_calibration_samples: int


Converter = Callable[[ConversionRequest], ComponentInfo]


class ConverterKey(NamedTuple):
    source_runtime: str
    target: str

    def __str__(self) -> str:
        return f"{self.source_runtime}/{self.target}"


_CONVERTERS: Dict[ConverterKey, Converter] = {}


def register_converter(source_runtime: str, target: str, target_runtime: Optional[str] = None) -> Callable:
    """Function decorator registering a converter of components on 'source_runtime' to 'target'.

    'target_runtime' is the runtime of the converted component. Only needed for targets this module doesn't know.
    """

    def decorator(converter: Converter) -> Converter:
        if target_runtime is not None:
            TARGET_RUNTIMES[target] = target_runtime
        if target not in TARGET_RUNTIMES:
            raise ValueError(f"Unknown conversion target '{target}'. Pass 'target_runtime' to register it.")
        _CONVERTERS[ConverterKey(source_runtime, target)] = converter
        return converter

    return decorator


def unregister_converter(source_runtime: str, target: str) -> None:
    _CONVERTERS.pop(ConverterKey(source_runtime, target), None)


def list_converters() -> List[ConverterKey]:
    """All available converters: registered ones and those of installed packages (not imported)."""
    return sorted(set(_CONVERTERS) | set(_entry_point_converters()))


def get_converter(source_runtime: str, target: str) -> Converter:
    key = ConverterKey(source_runtime, target)
    if key not in _CONVERTERS:
        entry_point = _entry_point_converters().get(key)
        if entry_point is None:
            raise KeyError(f"No converter '{key}'. Available: {[str(k) for k in list_converters()]}")
        try:
            converter = entry_point.load()
        except ImportError as e:
            raise ImportError(
                f"Converter '{key}' ('{entry_point.value}') can not be imported: {e}. Install its dependencies."
            ) from e
        _CONVERTERS.setdefault(key, converter)
    return _CONVERTERS[key]


def convert_bundle(
    path_source_bundle: PathLike,
    path_bundle: PathLike,
    target: str,
    options: Optional[Mapping[str, Any]] = None,
    calibration_inputs: Optional[Sequence[Mapping[str, Any]]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Convert every component of a bundle that supports 'target' and write the result as a new bundle.

    Args:
        path_source_bundle: Bundle to convert (zip or directory).
        path_bundle: Output bundle. A path ending with '.zip' creates a zip bundle, otherwise a directory.
        target: Conversion target, e.g. 'onnx', 'openvino' or 'openvino_int8'.
        options: Options for the target that override the ones stored with the components, e.g.
            '{"precision": "fp16"}' for 'openvino'. They don't apply to earlier steps of a conversion chain.
        calibration_inputs: Graph inputs of representative samples (e.g. '[{"image": path}, ...]'), for targets
            that calibrate (quantization). Each converted component is calibrated on what it gets at inference:
            the samples are run through the source graph up to that component.
        provenance: Extra provenance for the converted components, e.g. the calibration dataset.
        overwrite: Replace an existing output bundle.
    """
    if target not in TARGET_RUNTIMES:
        raise ValueError(f"Unknown conversion target '{target}'. Targets: {sorted(TARGET_RUNTIMES)}")

    with open_bundle(path_source_bundle) as path_source_dir, tempfile.TemporaryDirectory() as work_dir:
        model_info = ModelBundleInfo.from_json_file(path_source_dir / MODEL_INFO_FILENAME)
        chains = _conversion_chains(model_info, target, path_source_bundle)

        components: List[ComponentInfo] = []
        files: Dict[str, FileSource] = {}
        for component in model_info.graph.components:
            path_files_dir = path_source_dir
            for i_step, step_target in enumerate(chains.get(component.name, [])):
                path_output_dir = Path(work_dir) / component.name / str(i_step)
                path_output_dir.mkdir(parents=True)
                stored_options = component.conversions[step_target].options
                request = ConversionRequest(
                    component=component,
                    model_info=model_info,
                    path_source_dir=path_files_dir,
                    path_output_dir=path_output_dir,
                    target=step_target,
                    options={**stored_options, **(options or {} if step_target == target else {})},
                    calibration=_calibration_feed(path_source_bundle, component, calibration_inputs),
                    n_calibration_samples=len(calibration_inputs or []),
                )
                converted = get_converter(component.runtime, step_target)(request)
                _check_converted(component, converted, path_output_dir)
                sys_logger.info(
                    f"Converted component '{component.name}': {component.runtime} -> {step_target} "
                    f"(runtime '{converted.runtime}')"
                )
                if provenance and step_target == target:
                    merged_provenance = {**(converted.provenance or {}), **provenance}
                    converted = converted.model_copy(update={"provenance": merged_provenance})
                component, path_files_dir = converted, path_output_dir
            components.append(component)
            files.update({path: path_files_dir / path for path in component.files})

        graph = GraphInfo(inputs=model_info.graph.inputs, outputs=model_info.graph.outputs, components=components)
        converted_info = model_info.model_copy(update={"graph": graph})
        return write_bundle(path_bundle, converted_info, files, overwrite=overwrite)


def _conversion_chains(model_info: ModelBundleInfo, target: str, path_source_bundle: PathLike) -> Dict[str, List[str]]:
    """Component name -> targets to convert it to, in order, for all components that support 'target'."""
    chains: Dict[str, List[str]] = {}
    not_convertible: List[str] = []
    for component in model_info.graph.components:
        conversion = component.conversions.get(target)
        if conversion is None:
            continue
        if not conversion.supported:
            not_convertible.append(f"'{component.name}' does not support it: {conversion.reason or 'no reason given'}")
            continue
        chain = _find_chain(component, target)
        if chain is None:
            not_convertible.append(
                f"'{component.name}' (runtime '{component.runtime}') has no converter chain. Converters "
                f"(source runtime/target): {[str(key) for key in list_converters()]}"
            )
            continue
        chains[component.name] = chain

    for reason in not_convertible:
        sys_logger.warning(f"Not converting to '{target}': component {reason}")
    if not chains:
        details = " ".join(f"Component {reason}." for reason in not_convertible)
        supported = {c.name: sorted(c.conversions) for c in model_info.graph.components if c.conversions}
        raise ValueError(
            f"No component of '{path_source_bundle}' can be converted to '{target}'. {details} "
            f"Conversions listed by the components: {supported}"
        )
    return chains


def _find_chain(component: ComponentInfo, target: str) -> Optional[List[str]]:
    """Shortest list of conversion targets that takes the component from its runtime to 'target'.

    Every step needs a converter and has to be a supported conversion of the component.
    """
    converters = list_converters()
    supported = {name for name, conversion in component.conversions.items() if conversion.supported}
    queue: List[Tuple[str, List[str]]] = [(component.runtime, [])]
    visited = {component.runtime}
    while queue:
        runtime, chain = queue.pop(0)
        steps = [key.target for key in converters if key.source_runtime == runtime and key.target in supported]
        if target in steps:
            return [*chain, target]
        for step in steps:
            step_runtime = TARGET_RUNTIMES.get(step)  # Unknown until the converter of another package is imported
            if step_runtime is not None and step_runtime not in visited:
                visited.add(step_runtime)
                queue.append((step_runtime, [*chain, step]))
    return None


def _calibration_feed(
    path_source_bundle: PathLike,
    component: ComponentInfo,
    calibration_inputs: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[CalibrationFeed]:
    if not calibration_inputs:
        return None
    input_keys = list(component.inputs.values())
    graphs: List[ModelGraph] = []  # The graph up to the component, loaded on first use

    def feed() -> Iterator[Dict[str, Any]]:
        if not graphs:
            graphs.append(ModelGraph.load(path_source_bundle, outputs=input_keys))
        for sample_inputs in calibration_inputs:
            values = graphs[0].run(sample_inputs, outputs=input_keys)
            # A batched component gets its inputs with a batch axis, here a batch of one sample
            yield {
                port: np.asarray(values[key])[None] if component.batched else values[key]
                for port, key in component.inputs.items()
            }

    return feed


def _check_converted(source: ComponentInfo, converted: ComponentInfo, path_output_dir: Path) -> None:
    if converted.name != source.name or set(converted.outputs.values()) != set(source.outputs.values()):
        raise ValueError(
            f"A converter must keep the component's name and output keys. Source '{source.name}' writes "
            f"{sorted(source.outputs.values())}, converted '{converted.name}' {sorted(converted.outputs.values())}."
        )
    if set(converted.inputs.values()) != set(source.inputs.values()):
        raise ValueError(f"A converter must keep the input keys of component '{source.name}'.")
    missing = [path for path in converted.files if not (path_output_dir / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Converted component '{converted.name}' lists files that were not written: {missing}")


def _entry_point_converters() -> Dict[ConverterKey, EntryPoint]:
    converters = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        parts = entry_point.name.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Entry points of '{ENTRY_POINT_GROUP}' are named '<source runtime>/<target>', got "
                f"'{entry_point.name}'."
            )
        converters[ConverterKey(*parts)] = entry_point
    return converters
