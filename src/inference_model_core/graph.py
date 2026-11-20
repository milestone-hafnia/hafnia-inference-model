"""The model graph: loads the components of a bundle and runs them on a data dictionary.

All inputs, outputs and intermediate values are items of one dictionary. Each component reads the keys its
inputs are connected to and writes the keys of its outputs. Values are kept per sample, without a batch axis:
a 'batched' component gets the values of all samples stacked on a new first axis and its outputs are split
again, any other component is called once per sample. Components of different runtimes (NumPy, ONNX, OpenVINO,
torch) run in the same graph and exchange NumPy arrays.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

import numpy as np

from inference_model_core.bundle import MODEL_INFO_FILENAME, FileSource, PathLike, open_bundle, write_bundle
from inference_model_core.component import (
    Component,
    LoadOptions,
    call_component,
    find_component_class,
    get_component_class,
)
from inference_model_core.model_info import ComponentInfo, ModelBundleInfo


class ModelGraph:
    def __init__(self, model_info: ModelBundleInfo, components: Sequence[Component]):
        """
        Args:
            model_info: Model info of the bundle.
            components: Loaded components in graph order. A subset of the graph's components when the graph
                was loaded for some outputs only (see 'load').
        """
        self.model_info = model_info
        self.components = list(components)
        graph_order = [info.name for info in model_info.graph.components]
        loaded_order = [component.info.name for component in self.components]
        if loaded_order != [name for name in graph_order if name in set(loaded_order)]:
            raise ValueError(f"Components {loaded_order} are not in the order of the graph {graph_order}.")

    # --- storage -------------------------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path_bundle: PathLike,
        inference_engine: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        device: str = "CPU",
        num_threads: Optional[int] = None,
        engine_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
        outputs: Optional[Sequence[str]] = None,
    ) -> "ModelGraph":
        """Load a model bundle (zip or directory).

        Args:
            path_bundle: Zip or directory bundle.
            inference_engine: Engine for every component that has an implementation on it, e.g. 'openvino'
                runs the ONNX components of a bundle on OpenVINO. Other components run on their default engine.
            params: Overrides of component parameters. '{"score_threshold": 0.5}' applies to every component
                with that parameter, '{"nms.score_threshold": 0.5}' to component 'nms'. Overrides only apply
                to this instance, not to written bundles.
            device: Device for the engines, e.g. 'CPU'.
            num_threads: Threads per engine. 'None' uses the engine defaults.
            engine_options: Engine name -> settings only that engine knows, e.g.
                '{"openvino": {"performance_hint": "LATENCY"}}'.
            outputs: Only load the components needed to compute these data keys. By default all components.
        """
        options = LoadOptions(device=device, num_threads=num_threads, engine_options=engine_options or {})
        with open_bundle(path_bundle) as path_dir:
            model_info = ModelBundleInfo.from_json_file(path_dir / MODEL_INFO_FILENAME)
            graph = model_info.graph
            infos = graph.components_for(list(outputs)) if outputs is not None else graph.components
            classes = {info.name: _component_class(info, inference_engine) for info in infos}
            if inference_engine is not None and not any(c.engine == inference_engine for c in classes.values()):
                raise ValueError(
                    f"No component of '{path_bundle}' runs on engine '{inference_engine}'. Component runtimes: "
                    f"{sorted({info.runtime for info in infos})}."
                )
            unknown_engines = sorted(set(options.engine_options) - {c.engine for c in classes.values()})
            if unknown_engines:
                raise ValueError(f"'engine_options' are given for engines {unknown_engines}, which run no component.")
            param_overrides = _split_param_overrides(infos, classes, params or {})
            components = [
                classes[info.name].load(info, model_info, path_dir, options, params=param_overrides[info.name])
                for info in infos
            ]
        return cls(model_info, components)

    def write(self, path_bundle: PathLike, overwrite: bool = False) -> Path:
        """Write the model as a bundle. A path ending with '.zip' creates a zip bundle, otherwise a directory."""
        loaded = {component.info.name for component in self.components}
        not_loaded = [info.name for info in self.model_info.graph.components if info.name not in loaded]
        if not_loaded:
            raise ValueError(f"Can't write a partially loaded graph. Components {not_loaded} are not loaded.")
        files: Dict[str, FileSource] = {}
        for component in self.components:
            files.update(component.files())
        return write_bundle(path_bundle, self.model_info, files, overwrite=overwrite)

    # --- execution -----------------------------------------------------------------------------------
    def run(self, inputs: Mapping[str, Any], outputs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Run the graph on one sample: data dictionary with the graph inputs in, the graph outputs out."""
        results = self.run_batch({key: [value] for key, value in inputs.items()}, outputs=outputs)
        return {key: values[0] for key, values in results.items()}

    def run_batch(
        self,
        inputs: Mapping[str, Sequence[Any]],
        outputs: Optional[Sequence[str]] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[Any]]:
        """Run the graph on a batch of samples.

        Args:
            inputs: Graph input key -> one value per sample.
            outputs: Data keys to return, also intermediate ones. Only the components needed for them are run.
                By default the graph outputs.
            timings: If given, the seconds spent in each component are added to it under the component name.

        Returns:
            Output key -> one value per sample.
        """
        graph = self.model_info.graph
        missing, unknown = sorted(set(graph.inputs) - set(inputs)), sorted(set(inputs) - set(graph.inputs))
        if missing or unknown:
            raise KeyError(f"The graph has inputs {graph.inputs}. Missing: {missing}, unknown: {unknown}.")
        batch_sizes = {len(values) for values in inputs.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"All inputs need one value per sample, got lengths {sorted(batch_sizes)}.")
        (batch_size,) = batch_sizes

        output_keys = list(outputs) if outputs is not None else list(graph.outputs)
        needed = [info.name for info in graph.components_for(output_keys)]
        loaded = {component.info.name: component for component in self.components}
        not_loaded = [name for name in needed if name not in loaded]
        if not_loaded:
            raise ValueError(f"Outputs {output_keys} need components {not_loaded}, which are not loaded.")

        data: Dict[str, List[Any]] = {key: list(inputs[key]) for key in graph.inputs}
        for name in needed:
            start = time.perf_counter()
            data.update(_run_component(loaded[name], data, batch_size))
            if timings is not None:
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
        return {key: data[key] for key in output_keys}


def _component_class(info: ComponentInfo, inference_engine: Optional[str]) -> Type[Component]:
    if inference_engine is not None:
        component_class = find_component_class(info.type, info.runtime, inference_engine)
        if component_class is not None:
            return component_class
    return get_component_class(info.type, info.runtime)


def _split_param_overrides(
    infos: Sequence[ComponentInfo], classes: Mapping[str, Type[Component]], params: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Parameter overrides per component, from 'param' (all components with it) and 'component.param' keys."""
    overrides: Dict[str, Dict[str, Any]] = {info.name: {} for info in infos}
    parameters = {name: set(component_class.model_fields) for name, component_class in classes.items()}
    for key, value in params.items():
        component_name, _, param_name = key.rpartition(".")
        targets = (
            [component_name] if component_name else [name for name in parameters if param_name in parameters[name]]
        )
        if not targets or any(param_name not in parameters.get(name, set()) for name in targets):
            available = sorted(f"{name}.{param}" for name, fields in parameters.items() for param in fields)
            raise KeyError(f"No component parameter '{key}'. Parameters: {available}")
        for name in targets:
            overrides[name][param_name] = value
    return overrides


def _run_component(component: Component, data: Mapping[str, List[Any]], batch_size: int) -> Dict[str, List[Any]]:
    info = component.info
    results: Dict[str, List[Any]] = {key: [] for key in info.outputs.values()}

    def collect(outputs: Mapping[str, Any], n_samples: Optional[int]) -> None:
        missing = sorted(set(info.outputs) - set(outputs))
        if missing:
            raise KeyError(f"Component '{info.name}' returned no outputs {missing}. Returned: {sorted(outputs)}")
        for port, key in info.outputs.items():
            if n_samples is None:
                results[key].append(outputs[port])
                continue
            if len(outputs[port]) != n_samples:
                raise ValueError(
                    f"Output '{port}' of batched component '{info.name}' has {len(outputs[port])} entries on its "
                    f"first axis for a batch of {n_samples}."
                )
            results[key].extend(outputs[port][i] for i in range(n_samples))

    if not info.batched:
        for i in range(batch_size):
            inputs = {port: data[key][i] for port, key in info.inputs.items()}
            collect(call_component(component, inputs), n_samples=None)
        return results

    step = component.max_batch_size or max(batch_size, 1)
    for start in range(0, batch_size, step):
        stop = min(start + step, batch_size)
        inputs = {port: _stack(info, port, data[key][start:stop]) for port, key in info.inputs.items()}
        collect(call_component(component, inputs), n_samples=stop - start)
    return results


def _stack(info: ComponentInfo, port: str, values: Sequence[Any]) -> np.ndarray:
    try:
        return np.stack([np.asarray(value) for value in values])
    except ValueError as e:
        shapes = sorted({np.asarray(value).shape for value in values})
        raise ValueError(
            f"Can't batch input '{port}' of component '{info.name}': the samples have shapes {shapes}. Batched "
            "components need equally shaped values."
        ) from e
