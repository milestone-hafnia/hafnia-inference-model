"""Generic conversion of model bundles: the code behind 'ModelBundle.convert' and the registry of converters.

A bundle is converted component by component. Each component lists in its 'conversions' the targets it supports
(e.g. 'onnx', 'openvino', 'openvino_int8') with its export options. A converter turns a component of one type
into a target, which gives a component of another type (e.g. 'torchvision_ssd_network' -> target 'onnx' ->
'onnx_network'); chains such as torch -> onnx -> openvino are found automatically. Components that don't list
the target are copied unchanged, so e.g. image processing stays as it is next to a converted network.

Converters of installed packages are listed in the 'hafnia.inference_model_converters' entry point group, named
'<source type>/<target>', and are imported only when used. The source type '*' names a converter for every
component type that has no converter of its own for the target, e.g. '*/onnx' for all torch networks.
'register_converter' registers a converter of the running program, e.g. in a test.
"""

import tempfile
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from hafnia.log import sys_logger

from inference_model_core.bundle_io import FileSource, PathLike
from inference_model_core.definitions.network import ONNX_NETWORK_TYPE, OPENVINO_NETWORK_TYPE
from inference_model_core.graph import ModelGraph, load_graph
from inference_model_core.model_bundle import ComponentInfo, GraphInfo, ModelBundle

ENTRY_POINT_GROUP = "hafnia.inference_model_converters"

TARGET_ONNX = "onnx"
TARGET_OPENVINO = "openvino"
TARGET_OPENVINO_INT8 = "openvino_int8"
ANY_TYPE = "*"  # As the source type of a converter: every component type without a converter of its own
# Type of a component after it was converted to a target
TARGET_TYPES: Dict[str, str] = {
    TARGET_ONNX: ONNX_NETWORK_TYPE,
    TARGET_OPENVINO: OPENVINO_NETWORK_TYPE,
    TARGET_OPENVINO_INT8: OPENVINO_NETWORK_TYPE,
}

# Yields the inputs of the component being converted (port -> value), as it gets them at inference. Tensors are
# those of the graph engine that runs the source bundle
CalibrationFeed = Callable[[], Iterator[Dict[str, Any]]]


@dataclass(frozen=True)
class ConversionRequest:
    """What a converter gets.

    The converter writes the files of the converted component below 'path_output_dir' (in the component's
    folder) and returns its component info, with the same name and outputs as the source component.
    """

    component: ComponentInfo
    bundle: ModelBundle
    path_source_dir: Path  # Directory holding the files of 'component', at their bundle paths
    path_output_dir: Path
    target: str
    options: Dict[str, Any]  # The component's stored options for the target, merged with the caller's
    calibration: Optional[CalibrationFeed]  # 'None' when the caller gave no calibration inputs
    n_calibration_samples: int


Converter = Callable[[ConversionRequest], ComponentInfo]


class ConverterKey(NamedTuple):
    source_type: str
    target: str

    def __str__(self) -> str:
        return f"{self.source_type}/{self.target}"


_CONVERTERS: Dict[ConverterKey, Converter] = {}


def register_converter(source_type: str, target: str, target_type: Optional[str] = None) -> Callable:
    """Function decorator registering a converter of components of type 'source_type' to 'target'.

    'source_type' can be '*' (see the module docstring). 'target_type' is the type of the converted component:
    only needed for targets this module doesn't know.
    """

    def decorator(converter: Converter) -> Converter:
        if target_type is not None:
            TARGET_TYPES[target] = target_type
        if target not in TARGET_TYPES:
            raise ValueError(f"Unknown conversion target '{target}'. Pass 'target_type' to register it.")
        _CONVERTERS[ConverterKey(source_type, target)] = converter
        return converter

    return decorator


def unregister_converter(source_type: str, target: str) -> None:
    _CONVERTERS.pop(ConverterKey(source_type, target), None)


def list_converters() -> List[ConverterKey]:
    """All available converters: registered ones and those of installed packages (not imported)."""
    return sorted(set(_CONVERTERS) | set(_entry_point_converters()))


def get_converter(source_type: str, target: str) -> Converter:
    """The converter of 'source_type' to 'target', or the one of any type ('*') when the type has none."""
    available = list_converters()
    for key in (ConverterKey(source_type, target), ConverterKey(ANY_TYPE, target)):
        if key in available:
            return _load_converter(key)
    raise KeyError(f"No converter '{ConverterKey(source_type, target)}'. Available: {[str(k) for k in available]}")


def _load_converter(key: ConverterKey) -> Converter:
    if key not in _CONVERTERS:
        entry_point = _entry_point_converters()[key]
        try:
            converter = entry_point.load()
        except ImportError as e:
            raise ImportError(
                f"Converter '{key}' ('{entry_point.value}') can not be imported: {e}. Install its dependencies."
            ) from e
        _CONVERTERS.setdefault(key, converter)
    return _CONVERTERS[key]


def convert_model_bundle(
    bundle: ModelBundle,
    path_bundle: PathLike,
    target: str,
    options: Optional[Mapping[str, Any]] = None,
    calibration_inputs: Optional[Sequence[Mapping[str, Any]]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> ModelBundle:
    """Convert every component of 'bundle' that supports 'target' and write the result as a new bundle.

    See 'ModelBundle.convert', which calls this, for the arguments.
    """
    if target not in TARGET_TYPES:
        raise ValueError(f"Unknown conversion target '{target}'. Targets: {sorted(TARGET_TYPES)}")
    if bundle.path is None:
        raise ValueError(f"Model bundle '{bundle.name}' is only in memory. Write it before converting it.")
    path_source_bundle = bundle.path

    with bundle.open_dir() as path_source_dir, tempfile.TemporaryDirectory() as work_dir:
        chains = _conversion_chains(bundle, target, path_source_bundle)

        components: List[ComponentInfo] = []
        files: Dict[str, FileSource] = {}
        for component in bundle.graph.components:
            path_files_dir = path_source_dir
            for i_step, step_target in enumerate(chains.get(component.name, [])):
                path_output_dir = Path(work_dir) / component.name / str(i_step)
                path_output_dir.mkdir(parents=True)
                stored_options = component.conversions[step_target].options
                request = ConversionRequest(
                    component=component,
                    bundle=bundle,
                    path_source_dir=path_files_dir,
                    path_output_dir=path_output_dir,
                    target=step_target,
                    options={**stored_options, **(options or {} if step_target == target else {})},
                    calibration=_calibration_feed(bundle, component, calibration_inputs),
                    n_calibration_samples=len(calibration_inputs or []),
                )
                converted = get_converter(component.type, step_target)(request)
                _check_converted(component, converted, path_output_dir)
                sys_logger.info(
                    f"Converted component '{component.name}': '{component.type}' -> target '{step_target}' "
                    f"('{converted.type}')"
                )
                if provenance and step_target == target:
                    merged_provenance = {**(converted.provenance or {}), **provenance}
                    converted = converted.model_copy(update={"provenance": merged_provenance})
                component, path_files_dir = converted, path_output_dir
            components.append(component)
            files.update({path: path_files_dir / path for path in component.files})

        graph = GraphInfo(inputs=bundle.graph.inputs, outputs=bundle.graph.outputs, components=components)
        converted_bundle = bundle.model_copy(update={"graph": graph})
        return converted_bundle.write(path_bundle, files=files, overwrite=overwrite)


def _conversion_chains(bundle: ModelBundle, target: str, path_source_bundle: PathLike) -> Dict[str, List[str]]:
    """Component name -> targets to convert it to, in order, for all components that support 'target'."""
    chains: Dict[str, List[str]] = {}
    not_convertible: List[str] = []
    for component in bundle.graph.components:
        conversion = component.conversions.get(target)
        if conversion is None:
            continue
        if not conversion.supported:
            not_convertible.append(f"'{component.name}' does not support it: {conversion.reason or 'no reason given'}")
            continue
        chain = _find_chain(component, target)
        if chain is None:
            not_convertible.append(
                f"'{component.name}' (type '{component.type}') has no converter chain. Converters "
                f"(source type/target): {[str(key) for key in list_converters()]}"
            )
            continue
        chains[component.name] = chain

    for reason in not_convertible:
        sys_logger.warning(f"Not converting to '{target}': component {reason}")
    if not chains:
        details = " ".join(f"Component {reason}." for reason in not_convertible)
        supported = {c.name: sorted(c.conversions) for c in bundle.graph.components if c.conversions}
        raise ValueError(
            f"No component of '{path_source_bundle}' can be converted to '{target}'. {details} "
            f"Conversions listed by the components: {supported}"
        )
    return chains


def _find_chain(component: ComponentInfo, target: str) -> Optional[List[str]]:
    """Shortest list of conversion targets that takes the component from its type to 'target'.

    Every step needs a converter and has to be a supported conversion of the component.
    """
    converters = list_converters()
    supported = {name for name, conversion in component.conversions.items() if conversion.supported}
    queue: List[Tuple[str, List[str]]] = [(component.type, [])]
    visited = {component.type}
    while queue:
        component_type, chain = queue.pop(0)
        sources = (component_type, ANY_TYPE)
        steps = [key.target for key in converters if key.source_type in sources and key.target in supported]
        if target in steps:
            return [*chain, target]
        for step in steps:
            step_type = TARGET_TYPES.get(step)  # Unknown until the converter of another package is imported
            if step_type is not None and step_type not in visited:
                visited.add(step_type)
                queue.append((step_type, [*chain, step]))
    return None


def _calibration_feed(
    bundle: ModelBundle,
    component: ComponentInfo,
    calibration_inputs: Optional[Sequence[Mapping[str, Any]]],
) -> Optional[CalibrationFeed]:
    if not calibration_inputs:
        return None
    input_keys = list(component.inputs.values())
    graphs: List[ModelGraph] = []  # The graph up to the component, loaded on first use

    def feed() -> Iterator[Dict[str, Any]]:
        if not graphs:
            graphs.append(load_graph(bundle, outputs=input_keys))
        for sample_inputs in calibration_inputs:
            values = graphs[0].run(sample_inputs, outputs=input_keys)
            # A batched component gets its inputs with a batch axis, here a batch of one sample
            yield {
                port: values[key][None] if component.batched else values[key] for port, key in component.inputs.items()
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
                f"Entry points of '{ENTRY_POINT_GROUP}' are named '<source type>/<target>', got '{entry_point.name}'."
            )
        converters[ConverterKey(*parts)] = entry_point
    return converters
