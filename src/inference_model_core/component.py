"""Components: the steps of a model graph, and the registry of their implementations.

A component implementation is a 'Component' subclass for one (type, runtime, engine), e.g. ('network', 'onnx',
'onnxruntime') or ('resize_image', 'numpy', 'numpy'). Implementations register themselves with the
'register_component' class decorator. Separately installed packages list theirs in the
'hafnia.inference_model_components' entry point group, named '<type>/<runtime>/<engine>': they are imported only
when a graph needs them, so loading an ONNX bundle never imports torch.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, NamedTuple, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict

from inference_model_core.bundle import FileSource
from inference_model_core.model_info import (
    RUNTIME_NUMPY,
    RUNTIME_ONNX,
    RUNTIME_OPENVINO,
    RUNTIME_TORCH,
    ComponentInfo,
    ConversionInfo,
    ModelBundleInfo,
)

ENTRY_POINT_GROUP = "hafnia.inference_model_components"
BUILTIN_COMPONENTS_MODULE = "inference_model_core.components"

# Engine that runs a runtime unless the caller asks for another one
DEFAULT_ENGINES: Dict[str, str] = {
    RUNTIME_NUMPY: "numpy",
    RUNTIME_TORCH: "torch",
    RUNTIME_ONNX: "onnxruntime",
    RUNTIME_OPENVINO: "openvino",
}


class ComponentKey(NamedTuple):
    type: str
    runtime: str
    engine: str

    def __str__(self) -> str:
        return f"{self.type}/{self.runtime}/{self.engine}"


@dataclass(frozen=True)
class LoadOptions:
    """Load-time settings of the engines. They are not part of the model info."""

    device: str = "CPU"
    num_threads: Optional[int] = None
    # Engine name -> settings only that engine knows, e.g. {"openvino": {"performance_hint": "LATENCY"}}
    engine_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def for_engine(self, engine: str) -> Dict[str, Any]:
        return dict(self.engine_options.get(engine, {}))


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


T = TypeVar("T", bound="Component")


class Component(ABC):
    """Base class of all component implementations.

    Subclasses set 'type', 'runtime' and 'engine', their typed 'params_model' and their ports, and implement
    'run'. Components with files also implement 'load' and 'files'.
    """

    type: ClassVar[str]
    runtime: ClassVar[str]
    engine: ClassVar[str]
    params_model: ClassVar[Type[BaseModel]] = NoParams
    # Port names. Components with ports that depend on their parameters (e.g. tensor names) override 'check_ports'.
    input_ports: ClassVar[Tuple[str, ...]] = ()
    optional_input_ports: ClassVar[Tuple[str, ...]] = ()
    output_ports: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, info: ComponentInfo, model_info: ModelBundleInfo):
        cls = type(self)
        if (info.type, info.runtime) != (cls.type, cls.runtime):
            raise ValueError(
                f"'{cls.__name__}' implements type '{cls.type}' on runtime '{cls.runtime}', but component "
                f"'{info.name}' has type '{info.type}' on runtime '{info.runtime}'."
            )
        self.info = info
        self.model_info = model_info
        self.params: Any = cls.params_model.model_validate(info.params)
        self.check_ports()

    @classmethod
    def key(cls) -> ComponentKey:
        return ComponentKey(type=cls.type, runtime=cls.runtime, engine=cls.engine)

    def check_ports(self) -> None:
        """The component info must connect all required inputs and only known ports."""
        cls = type(self)
        connected_inputs, connected_outputs = set(self.info.inputs), set(self.info.outputs)
        missing = sorted(set(cls.input_ports) - connected_inputs)
        unknown_inputs = sorted(connected_inputs - set(cls.input_ports) - set(cls.optional_input_ports))
        unknown_outputs = sorted(connected_outputs - set(cls.output_ports))
        if missing or unknown_inputs or unknown_outputs:
            raise ValueError(
                f"Component '{self.info.name}' ('{cls.type}') has inputs {list(cls.input_ports)}, optional inputs "
                f"{list(cls.optional_input_ports)} and outputs {list(cls.output_ports)}. Missing inputs: {missing}, "
                f"unknown inputs: {unknown_inputs}, unknown outputs: {unknown_outputs}."
            )

    # --- storage -------------------------------------------------------------------------------------
    @classmethod
    def load(cls: Type[T], info: ComponentInfo, model_info: ModelBundleInfo, path_dir: Path, options: LoadOptions) -> T:
        """Create the component of a bundle extracted at 'path_dir'. The default is for components without
        files. Files have to be read here: 'path_dir' can be a temporary directory."""
        if info.files:
            raise NotImplementedError(f"'{cls.__name__}' has files and must implement 'load'.")
        return cls(info, model_info)

    def files(self) -> Dict[str, FileSource]:
        """The component's bundle files (path relative to the bundle -> content), for writing the bundle."""
        return {}

    # --- execution -----------------------------------------------------------------------------------
    @property
    def max_batch_size(self) -> Optional[int]:
        """Largest batch a batched component takes in one call. 'None' for no limit."""
        return None

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the outputs (port -> value) from the connected inputs (port -> value).

        Values are for one sample, or for a batch stacked on a new first axis when the component is 'batched'.
        """

    # --- building graphs -----------------------------------------------------------------------------
    @classmethod
    def create_info(
        cls,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        params: Union[BaseModel, Dict[str, Any], None] = None,
        files: Optional[List[str]] = None,
        batched: bool = False,
        conversions: Optional[Dict[str, ConversionInfo]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ComponentInfo:
        """Component info of this implementation, with validated parameters."""
        params_dict = params.model_dump(mode="json") if isinstance(params, BaseModel) else dict(params or {})
        validated_params = cls.params_model.model_validate(params_dict)
        return ComponentInfo(
            name=name,
            type=cls.type,
            runtime=cls.runtime,
            inputs=inputs,
            outputs=outputs,
            params=validated_params.model_dump(mode="json"),
            files=files or [],
            batched=batched,
            conversions=conversions or {},
            provenance=provenance,
        )


class NumpyComponent(Component):
    """Base class of components written with NumPy, which need no engine."""

    runtime = RUNTIME_NUMPY
    engine = RUNTIME_NUMPY


_COMPONENTS: Dict[ComponentKey, Type[Component]] = {}
_builtin_loaded = False

C = TypeVar("C", bound=Type[Component])


def register_component(component_class: C) -> C:
    """Class decorator registering an implementation under its (type, runtime, engine) key."""
    key = component_class.key()
    registered_class = _COMPONENTS.get(key)
    is_same_class = registered_class is not None and (
        registered_class.__module__ == component_class.__module__
        and registered_class.__qualname__ == component_class.__qualname__
    )
    if registered_class is not None and not is_same_class:
        raise ValueError(
            f"Component '{key}' is already registered by '{registered_class.__qualname__}', "
            f"can not register '{component_class.__qualname__}'."
        )
    _COMPONENTS[key] = component_class
    return component_class


def unregister_component(key: ComponentKey) -> None:
    _COMPONENTS.pop(key, None)


def list_components() -> List[ComponentKey]:
    """All available implementations: registered ones and those of installed packages (not imported)."""
    _load_builtin_components()
    return sorted(set(_COMPONENTS) | set(_entry_point_components()))


def find_component_class(component_type: str, runtime: str, engine: str) -> Optional[Type[Component]]:
    """The implementation of (type, runtime) on 'engine', or 'None' if there is none.

    Raises 'ImportError' when the implementation exists but its dependencies are not installed.
    """
    _load_builtin_components()
    key = ComponentKey(type=component_type, runtime=runtime, engine=engine)
    if key in _COMPONENTS:
        return _COMPONENTS[key]

    entry_point = _entry_point_components().get(key)
    if entry_point is None:
        return None
    try:
        component_class = entry_point.load()
    except ImportError as e:
        raise ImportError(
            f"Component '{key}' ('{entry_point.value}') can not be imported: {e}. Install the dependencies of "
            f"the '{engine}' engine."
        ) from e
    if not (isinstance(component_class, type) and issubclass(component_class, Component)):
        raise TypeError(f"Entry point '{entry_point.name}' ('{entry_point.value}') is not a 'Component' class.")
    if component_class.key() != key:
        raise ValueError(f"Entry point '{entry_point.name}' names '{component_class.key()}' ('{entry_point.value}').")
    return register_component(component_class)


def get_component_class(component_type: str, runtime: str, engine: Optional[str] = None) -> Type[Component]:
    """The implementation of (type, runtime) on 'engine', by default the runtime's default engine."""
    engine = engine or DEFAULT_ENGINES.get(runtime, runtime)
    component_class = find_component_class(component_type, runtime, engine)
    if component_class is None:
        available = "\n".join(f"  - {key}" for key in list_components()) or "  (none)"
        raise KeyError(
            f"No component is registered for type '{component_type}' on runtime '{runtime}' with engine '{engine}'.\n"
            f"Available components (type/runtime/engine):\n{available}\n"
            "Install the package providing it, or import the module defining it."
        )
    return component_class


def _load_builtin_components() -> None:
    global _builtin_loaded
    if not _builtin_loaded:
        _builtin_loaded = True
        importlib.import_module(BUILTIN_COMPONENTS_MODULE)


def _entry_point_components() -> Dict[ComponentKey, EntryPoint]:
    components = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        parts = entry_point.name.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"Entry points of '{ENTRY_POINT_GROUP}' are named '<type>/<runtime>/<engine>', got "
                f"'{entry_point.name}'."
            )
        components[ComponentKey(*parts)] = entry_point
    return components
