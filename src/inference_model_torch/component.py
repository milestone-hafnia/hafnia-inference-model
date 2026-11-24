"""Component implementations of the torch graph engine, and their registry.

An implementation computes one component definition ('inference_model_core.definition'). It gets the
definition with the component's parameters as 'self.params' and implements 'forward', whose parameters are the
input ports of the definition. It returns the outputs by port name:

    @register_component
    class TorchImageToTensor(Component[ImageToTensor]):
        def forward(self, image: torch.Tensor) -> Dict[str, Any]:
            return {"tensor": image_to_tensor(image, scale=self.params.scale, ...)}

Components exchange 'torch.Tensor's. Most are written in torch and compute on the device of the graph. Networks
stored as ONNX or OpenVINO IR files run on their own inference engine and still take and return torch tensors
(see 'inference_model_torch.network').

A definition has one default implementation and can have alternatives on other inference engines, chosen with
'inference_engine' when loading a graph (e.g. an ONNX network on OpenVINO instead of ONNX Runtime).
Implementations register with the 'register_component' class decorator. Separately installed packages list
theirs in the 'hafnia.inference_model_components' entry point group, named '<type>' for the default
implementation and '<type>/<inference engine>' for an alternative: they are imported only when a graph needs
them, so loading an ONNX bundle never imports OpenVINO.
"""

import importlib
import inspect
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Generic, List, Mapping, Optional, Type, TypeVar, Union, overload

import numpy as np
import torch

from inference_model_core.bundle_io import FileSource
from inference_model_core.definition import ComponentDefinition
from inference_model_core.model_bundle import ComponentInfo, ModelBundle

GRAPH_ENGINE = "torch"
ENGINE = "torch"  # Inference engine of the components written in torch
ENTRY_POINT_GROUP = "hafnia.inference_model_components"
BUILTIN_COMPONENTS_MODULE = "inference_model_torch.components"


@dataclass(frozen=True)
class LoadOptions:
    """Load-time settings of the inference engines. They are not part of the model info."""

    device: str = "CPU"
    num_threads: Optional[int] = None
    # Inference engine name -> settings only that engine knows, e.g. {"openvino": {"performance_hint": "LATENCY"}}
    engine_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def for_engine(self, engine: str) -> Dict[str, Any]:
        return dict(self.engine_options.get(engine, {}))


@dataclass(frozen=True)
class ComponentContext:
    """The place of a component in a loaded graph."""

    info: ComponentInfo  # Name, connections and files of the component
    bundle: ModelBundle
    # Directory with the extracted bundle. It can be temporary: files have to be read in 'Component.setup'
    path_dir: Optional[Path] = None
    options: LoadOptions = field(default_factory=LoadOptions)


D = TypeVar("D", bound=ComponentDefinition)
T = TypeVar("T", bound="Component")


class Component(ABC, Generic[D]):
    """Base class of the implementations: 'Component[<definition>]' names the definition it computes.

    Subclasses implement 'forward'. Implementations that need more than their parameters to run (model files, a
    task of the model info) get it in 'setup'.
    """

    definition: ClassVar[Type[ComponentDefinition]]
    engine: ClassVar[str] = ENGINE

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # 'class TorchResizeImage(Component[ResizeImage])' sets 'definition = ResizeImage'
        for base in getattr(cls, "__orig_bases__", ()):
            arguments = typing.get_args(base)
            if arguments and isinstance(arguments[0], type) and issubclass(arguments[0], ComponentDefinition):
                cls.definition = arguments[0]

    def __init__(self, params: D, context: ComponentContext):
        """
        Args:
            params: The definition with the parameters of the component.
            context: The place of the component in the graph.
        """
        params.check_connections(context.info)
        self.params = params
        self.context = context
        self.setup()

    @classmethod
    def load(
        cls: Type[T],
        info: ComponentInfo,
        bundle: ModelBundle,
        path_dir: Optional[Path] = None,
        options: Optional[LoadOptions] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> T:
        """Create the implementation of a component of a model info and set it up to run.

        Args:
            info: Component info of the model info.
            bundle: The model bundle of the component.
            path_dir: Directory with the extracted bundle, for components with files.
            options: Load-time settings of the inference engines.
            params: Overrides of the parameters in 'info'.
        """
        context = ComponentContext(info=info, bundle=bundle, path_dir=path_dir, options=options or LoadOptions())
        return cls(cls.definition.from_info(info, params), context)

    @classmethod
    def key(cls) -> str:
        """'<type>/<inference engine>', e.g. 'resize_image/torch' or 'onnx_network/onnxruntime'."""
        return f"{cls.definition.type}/{cls.engine}"

    @property
    def info(self) -> ComponentInfo:
        return self.context.info

    @property
    def device(self) -> torch.device:
        """Device the component takes its input tensors on: the graph moves them there before 'forward'."""
        return torch_device(self.context.options.device)

    # --- implementation ------------------------------------------------------------------------------
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Compute the outputs from the inputs.

        Subclasses name the input ports of their definition as parameters (optional ports with default 'None')
        and return the outputs by port name. Values are for one sample, or for a batch stacked on a new first
        axis when the component is connected as 'batched'. Tensors are 'torch.Tensor's on 'self.device'.
        """

    def setup(self) -> None:
        """Prepare for running, with 'self.params' and 'self.context': read the component's files, look up a
        task. Called once, when the graph is loaded."""

    def files(self) -> Dict[str, FileSource]:
        """The component's bundle files (path relative to the bundle -> content), for writing the bundle."""
        return {}

    @property
    def max_batch_size(self) -> Optional[int]:
        """Largest batch a batched component takes in one call. 'None' for no limit."""
        return None


def torch_device(device: str) -> torch.device:
    """Torch device for a device name of the load options.

    'CPU', 'CUDA' and torch device names (e.g. 'cuda:1') are used as they are. Other names belong to another
    inference engine (e.g. OpenVINO's 'GPU', 'NPU' or 'AUTO'): the components written in torch then run on the CPU.
    """
    try:
        return torch.device(device.lower())
    except RuntimeError:
        return torch.device("cpu")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """A tensor as a C-contiguous NumPy array for an inference engine, without a copy when it is on the CPU."""
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


def from_numpy(array: Any) -> torch.Tensor:
    """The output of an inference engine as a tensor, without a copy when the array is writable."""
    array = np.asarray(array)
    return torch.from_numpy(array if array.flags.writeable else array.copy())


# --- registry ----------------------------------------------------------------------------------------
_DEFAULTS: Dict[str, Type[Component]] = {}  # type -> default implementation
_ALTERNATIVES: Dict[str, Type[Component]] = {}  # '<type>/<inference engine>' -> implementation
_builtin_loaded = False

C = TypeVar("C", bound=Type[Component])


@overload
def register_component(component_class: C) -> C: ...
@overload
def register_component(*, alternative: bool) -> Callable[[C], C]: ...
def register_component(component_class: Optional[C] = None, *, alternative: bool = False) -> Union[C, Callable[[C], C]]:
    """Class decorator registering an implementation of its definition.

    '@register_component' registers the default implementation of the definition's type.
    '@register_component(alternative=True)' registers an implementation on another inference engine, used when
    a graph is loaded with 'inference_engine=<the engine of the class>'.
    """

    def register(component_class: C) -> C:
        if not isinstance(getattr(component_class, "definition", None), type):
            raise TypeError(f"'{component_class.__name__}' names no definition: subclass 'Component[<definition>]'.")
        _check_forward_signature(component_class)
        registry = _ALTERNATIVES if alternative else _DEFAULTS
        name = component_class.key() if alternative else component_class.definition.type
        registered_class = registry.get(name)
        is_same_class = registered_class is not None and (
            registered_class.__module__ == component_class.__module__
            and registered_class.__qualname__ == component_class.__qualname__
        )
        if registered_class is not None and not is_same_class:
            raise ValueError(
                f"Component '{name}' is already registered by '{registered_class.__qualname__}', "
                f"can not register '{component_class.__qualname__}'."
            )
        registry[name] = component_class
        return component_class

    return register(component_class) if component_class is not None else register


def unregister_component(component_class: Type[Component]) -> None:
    for registry in (_DEFAULTS, _ALTERNATIVES):
        for name in [name for name, registered_class in registry.items() if registered_class is component_class]:
            del registry[name]


def list_components() -> List[str]:
    """All available implementations, registered ones and those of installed packages (not imported): '<type>'
    for the default implementation of a type, '<type>/<inference engine>' for an alternative."""
    _load_builtin_components()
    return sorted({*_DEFAULTS, *_ALTERNATIVES, *_entry_point_components()})


def get_component_class(component_type: str) -> Type[Component]:
    """The default implementation of a component type."""
    component_class = _find_component_class(component_type, _DEFAULTS)
    if component_class is None:
        available = "\n".join(f"  - {name}" for name in list_components()) or "  (none)"
        raise KeyError(
            f"The torch graph engine has no implementation of component type '{component_type}'.\n"
            f"Available implementations ('<type>' or '<type>/<inference engine>'):\n{available}\n"
            "Install the package providing it, or import the module defining it."
        )
    return component_class


def find_alternative_class(component_type: str, engine: str) -> Optional[Type[Component]]:
    """The implementation of a component type on the inference engine 'engine', if it is not the default one."""
    return _find_component_class(f"{component_type}/{engine}", _ALTERNATIVES)


def _find_component_class(name: str, registry: Dict[str, Type[Component]]) -> Optional[Type[Component]]:
    """Raises 'ImportError' when the implementation exists but its dependencies are not installed."""
    _load_builtin_components()
    if name in registry:
        return registry[name]
    entry_point = _entry_point_components().get(name)
    if entry_point is None:
        return None
    try:
        component_class = entry_point.load()
    except ImportError as e:
        raise ImportError(
            f"Component '{name}' ('{entry_point.value}') can not be imported: {e}. Install the dependencies of "
            "its inference engine."
        ) from e
    if not (isinstance(component_class, type) and issubclass(component_class, Component)):
        raise TypeError(f"Entry point '{entry_point.name}' ('{entry_point.value}') is not a 'Component' class.")
    is_alternative = registry is _ALTERNATIVES
    if name != (component_class.key() if is_alternative else component_class.definition.type):
        raise ValueError(f"Entry point '{entry_point.name}' names '{component_class.key()}' ('{entry_point.value}').")
    return register_component(alternative=is_alternative)(component_class)


def _check_forward_signature(component_class: Type[Component]) -> None:
    """The parameters of 'forward' must be the input ports of the definition, so the two can't drift apart.
    Skipped for definitions with configured ports (e.g. tensor names), whose 'forward' takes '**tensors'."""
    definition = component_class.definition
    parameters = list(inspect.signature(component_class.forward).parameters.values())[1:]
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        return
    required = {parameter.name for parameter in parameters if parameter.default is parameter.empty}
    optional = {parameter.name for parameter in parameters if parameter.default is None}
    declared_required = {port.name for port in definition.input_ports if not port.optional}
    declared_optional = {port.name for port in definition.input_ports if port.optional}
    if (required, optional) != (declared_required, declared_optional) or len(parameters) != len(required | optional):
        raise TypeError(
            f"'{component_class.__name__}.forward' takes {[str(parameter) for parameter in parameters]}, but "
            f"'{definition.__name__}' declares the inputs {sorted(declared_required)} and the optional inputs "
            f"{sorted(declared_optional)} (parameters with default 'None')."
        )


def _load_builtin_components() -> None:
    global _builtin_loaded
    if not _builtin_loaded:
        _builtin_loaded = True
        importlib.import_module(BUILTIN_COMPONENTS_MODULE)


def _entry_point_components() -> Dict[str, EntryPoint]:
    components = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if not 1 <= len(entry_point.name.split("/")) <= 2:
            raise ValueError(
                f"Entry points of '{ENTRY_POINT_GROUP}' are named '<type>' or '<type>/<inference engine>', got "
                f"'{entry_point.name}'."
            )
        components[entry_point.name] = entry_point
    return components
