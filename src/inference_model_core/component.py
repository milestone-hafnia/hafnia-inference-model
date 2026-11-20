"""Components: the steps of a model graph, and the registry of their implementations.

A component is a pydantic model: its fields are its configuration (stored as 'params' in the model info) and its
'forward' method is the implementation. The signature of 'forward' names the ports: the parameters are the
inputs and the fields of the returned 'NamedTuple' the outputs.

    class ResizeImage(NumpyComponent):
        type = "resize_image"
        height: int
        width: int

        def forward(self, image: np.ndarray) -> ResizedImage:  # 'ResizedImage(image, transform)'
            ...

An implementation is a 'Component' subclass for one (type, runtime, engine), e.g. ('network', 'onnx',
'onnxruntime') or ('resize_image', 'numpy', 'numpy'). Implementations register themselves with the
'register_component' class decorator. Separately installed packages list theirs in the
'hafnia.inference_model_components' entry point group, named '<type>/<runtime>/<engine>': they are imported only
when a graph needs them, so loading an ONNX bundle never imports torch.
"""

import functools
import importlib
import inspect
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, NamedTuple, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

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


@dataclass(frozen=True)
class ComponentContext:
    """The place of a component in a loaded graph."""

    info: ComponentInfo  # Name, connections and files of the component
    model_info: ModelBundleInfo
    # Directory with the extracted bundle. It can be temporary: files have to be read in 'Component.setup'
    path_dir: Optional[Path] = None
    options: LoadOptions = field(default_factory=LoadOptions)


class Ports(NamedTuple):
    inputs: Tuple[str, ...]
    optional_inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


T = TypeVar("T", bound="Component")


class Component(BaseModel, ABC):
    """Base class of all component implementations.

    Subclasses set 'type', 'runtime' and 'engine', declare their configuration as fields and implement
    'forward'. Components that need more than their configuration to run (model files, a task of the model
    info) get it in 'setup'.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ClassVar[str]
    runtime: ClassVar[str]
    engine: ClassVar[str]

    _context: Optional[ComponentContext] = PrivateAttr(default=None)

    @classmethod
    def key(cls) -> ComponentKey:
        return ComponentKey(type=cls.type, runtime=cls.runtime, engine=cls.engine)

    # --- implementation ------------------------------------------------------------------------------
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the outputs from the inputs.

        Subclasses name their input ports as parameters (a parameter with default 'None' is an optional port)
        and return a 'NamedTuple' whose fields are the output ports. Values are for one sample, or for a batch
        stacked on a new first axis when the component is connected as 'batched'.
        """

    def ports(self) -> Ports:
        """The ports of the component, by default read from the signature of 'forward'. Components with ports
        that depend on their configuration (e.g. tensor names) override this."""
        return signature_ports(type(self))

    def setup(self) -> None:
        """Prepare for running in a graph, with 'self.context': read the component's files, look up a task.
        Called once, when the graph is loaded."""

    def files(self) -> Dict[str, FileSource]:
        """The component's bundle files (path relative to the bundle -> content), for writing the bundle."""
        return {}

    @property
    def max_batch_size(self) -> Optional[int]:
        """Largest batch a batched component takes in one call. 'None' for no limit."""
        return None

    # --- in a graph ----------------------------------------------------------------------------------
    def connect(
        self,
        name: str,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        files: Optional[List[str]] = None,
        batched: bool = False,
        conversions: Optional[Dict[str, ConversionInfo]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ComponentInfo:
        """The component as a step of a graph: its component info for the model info.

        Args:
            name: Unique name in the graph.
            inputs: Input port -> data dictionary key it reads.
            outputs: Output port -> data dictionary key it writes.
            files: Files of the component, relative to the bundle.
            batched: Call the component once per batch instead of once per sample.
            conversions: Conversion target -> support and export options.
            provenance: Where the component comes from.
        """
        info = ComponentInfo(
            name=name,
            type=type(self).type,
            runtime=type(self).runtime,
            inputs=inputs,
            outputs=outputs,
            params=self.model_dump(mode="json"),
            files=files or [],
            batched=batched,
            conversions=conversions or {},
            provenance=provenance,
        )
        self.check_connections(info)
        return info

    @classmethod
    def load(
        cls: Type[T],
        info: ComponentInfo,
        model_info: ModelBundleInfo,
        path_dir: Optional[Path] = None,
        options: Optional[LoadOptions] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> T:
        """Create the component of a model info and set it up to run.

        Args:
            info: Component info of the model info.
            model_info: The model info.
            path_dir: Directory with the extracted bundle, for components with files.
            options: Load-time settings of the engines.
            params: Overrides of the configuration in 'info'.
        """
        if (info.type, info.runtime) != (cls.type, cls.runtime):
            raise ValueError(
                f"'{cls.__name__}' implements type '{cls.type}' on runtime '{cls.runtime}', but component "
                f"'{info.name}' has type '{info.type}' on runtime '{info.runtime}'."
            )
        component = cls.model_validate({**info.params, **(params or {})})
        context = ComponentContext(
            info=info, model_info=model_info, path_dir=path_dir, options=options or LoadOptions()
        )
        return component.bind(context)

    def bind(self: T, context: ComponentContext) -> T:
        """Give the component its place in a graph and set it up."""
        self.check_connections(context.info)
        self._context = context
        self.setup()
        return self

    @property
    def context(self) -> ComponentContext:
        if self._context is None:
            raise RuntimeError(f"This '{type(self).type}' component is not part of a loaded graph (see 'load').")
        return self._context

    @property
    def info(self) -> ComponentInfo:
        return self.context.info

    def check_connections(self, info: ComponentInfo) -> None:
        """The component info must connect all required inputs and only known ports."""
        ports = self.ports()
        missing = sorted(set(ports.inputs) - set(info.inputs))
        unknown_inputs = sorted(set(info.inputs) - set(ports.inputs) - set(ports.optional_inputs))
        unknown_outputs = sorted(set(info.outputs) - set(ports.outputs))
        if missing or unknown_inputs or unknown_outputs:
            raise ValueError(
                f"Component '{info.name}' ('{type(self).type}') has inputs {list(ports.inputs)}, optional inputs "
                f"{list(ports.optional_inputs)} and outputs {list(ports.outputs)}. Missing inputs: {missing}, "
                f"unknown inputs: {unknown_inputs}, unknown outputs: {unknown_outputs}."
            )


class NumpyComponent(Component):
    """Base class of components written with NumPy, which need no engine."""

    runtime = RUNTIME_NUMPY
    engine = RUNTIME_NUMPY


@functools.lru_cache(maxsize=None)
def signature_ports(component_class: Type[Component]) -> Ports:
    """Ports of a component class from the signature of its 'forward': the parameters are the inputs (optional
    with default 'None') and the fields of the returned 'NamedTuple' the outputs."""
    name = f"'{component_class.__name__}.forward'"
    inputs, optional_inputs = [], []
    for parameter in list(inspect.signature(component_class.forward).parameters.values())[1:]:
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            raise TypeError(
                f"{name} takes '{parameter}'. Input ports are named parameters; components with other ports "
                "override 'ports'."
            )
        if parameter.default is parameter.empty:
            inputs.append(parameter.name)
        elif parameter.default is None:
            optional_inputs.append(parameter.name)
        else:
            raise TypeError(f"{name}: the default of the optional input '{parameter.name}' must be 'None'.")

    return_type = typing.get_type_hints(component_class.forward).get("return")
    outputs = getattr(return_type, "_fields", None)
    if not (isinstance(return_type, type) and issubclass(return_type, tuple) and outputs):
        raise TypeError(f"{name} must be annotated to return a 'NamedTuple'. Its fields are the output ports.")
    return Ports(inputs=tuple(inputs), optional_inputs=tuple(optional_inputs), outputs=tuple(outputs))


def call_component(component: Component, inputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Call 'forward' with the inputs by port name. Returns the outputs by port name."""
    outputs = component.forward(**inputs)
    if isinstance(outputs, Mapping):  # Components with configured ports, e.g. networks by tensor name
        return dict(outputs)
    if hasattr(outputs, "_asdict"):
        return outputs._asdict()
    raise TypeError(
        f"'{type(component).__name__}.forward' returned a '{type(outputs).__name__}'. Components return a "
        "'NamedTuple' (or a dictionary by port name)."
    )


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
