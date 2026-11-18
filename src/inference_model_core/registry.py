"""Flavor registry and 'inference_model_from_path'.

Flavors register themselves with the 'register_flavor' class decorator when their module is imported.
Modules of separately installed packages are imported through the 'hafnia.inference_models' entry point
group, so installing e.g. 'inference-model-openvino' makes its flavors available without code changes.
"""

from importlib.metadata import entry_points
from typing import Any, Dict, List, Optional, Type, TypeVar

from hafnia.log import sys_logger

from inference_model_core.bundle import PathLike, read_model_info
from inference_model_core.inference_model import InferenceModelBase
from inference_model_core.model_info import FlavorKey

ENTRY_POINT_GROUP = "hafnia.inference_models"

_FLAVORS: Dict[FlavorKey, Type[InferenceModelBase]] = {}
_entry_points_loaded = False

T = TypeVar("T", bound=Type[InferenceModelBase])


def register_flavor(flavor_class: T) -> T:
    """Class decorator registering a flavor under its (inference_engine, framework, decoder) key."""
    key = flavor_class.flavor_key()
    registered_class = _FLAVORS.get(key)
    is_same_class = registered_class is not None and (
        registered_class.__module__ == flavor_class.__module__
        and registered_class.__qualname__ == flavor_class.__qualname__
    )
    if registered_class is not None and not is_same_class:
        raise ValueError(
            f"Flavor key {key} is already registered by '{registered_class.__qualname__}', "
            f"can not register '{flavor_class.__qualname__}'."
        )
    _FLAVORS[key] = flavor_class
    return flavor_class


def unregister_flavor(key: FlavorKey) -> None:
    _FLAVORS.pop(key, None)


def load_entry_point_flavors() -> None:
    """Import all modules registered in the entry point group. Done once per process."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            entry_point.load()
        except ImportError as e:  # E.g. optional engine dependencies are not installed
            sys_logger.warning(f"Skipping inference model flavors '{entry_point.value}': {e}")


def list_flavors() -> List[FlavorKey]:
    load_entry_point_flavors()
    return sorted(_FLAVORS)


def get_flavor(key: FlavorKey) -> Type[InferenceModelBase]:
    load_entry_point_flavors()
    if key not in _FLAVORS:
        available = "\n".join(f"  - {flavor_key}" for flavor_key in sorted(_FLAVORS)) or "  (none)"
        raise KeyError(
            f"No inference model flavor is registered for {key}.\n"
            f"Available flavors:\n{available}\n"
            f"Install the package providing the '{key.inference_engine}' engine flavors, "
            f"or import the module defining the flavor."
        )
    return _FLAVORS[key]


def inference_model_from_path(
    path_bundle: PathLike,
    inference_engine: Optional[str] = None,
    postprocessing_params: Optional[Dict[str, Any]] = None,
    **engine_options: Any,
) -> InferenceModelBase:
    """Load a model bundle with the flavor matching its model info.

    Args:
        path_bundle: Zip or directory bundle.
        inference_engine: Engine to run the model on. Defaults to the bundle's default engine. Any engine
            with a flavor supporting the bundle's model format works, e.g. an ONNX bundle on OpenVINO.
        postprocessing_params: Overrides of the decoder parameters, e.g. '{"score_threshold": 0.5}'.
        engine_options: Engine specific load-time settings, e.g. 'device="CPU"'.
    """
    model_info = read_model_info(path_bundle)
    flavor_class = get_flavor(model_info.flavor_key(inference_engine))
    if model_info.runtime.model_format not in flavor_class.supported_formats:
        raise ValueError(
            f"Engine '{flavor_class.inference_engine}' ('{flavor_class.__name__}') can not run model format "
            f"'{model_info.runtime.model_format}'. Supported formats: {list(flavor_class.supported_formats)}."
        )
    return flavor_class.load_model(path_bundle, postprocessing_params=postprocessing_params, **engine_options)
