"""The torch graph engine: loads the components of a bundle and runs them on a data dictionary.

Implements 'inference_model_core.ModelGraph'. Values are kept per sample, without a batch axis: a 'batched'
component gets the tensors of all samples stacked on a new first axis and its outputs are split again, any other
component is called once per sample. Tensors are 'torch.Tensor's and stay on the device of the component that
produced them: the graph moves them only when the next component needs them elsewhere, so a graph of torch
components runs on the GPU from the image to the detections.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

import torch

from inference_model_core.bundle_io import FileSource, PathLike
from inference_model_core.model_bundle import ComponentInfo, ModelBundle, write_by_suffix
from inference_model_torch.component import Component, LoadOptions, find_alternative_class, get_component_class


class TorchModelGraph:
    def __init__(self, bundle: ModelBundle, components: Sequence[Component]):
        """
        Args:
            bundle: The model bundle the components are loaded from.
            components: Loaded components in graph order. A subset of the graph's components when the graph
                was loaded for some outputs only (see 'load').
        """
        self.bundle = bundle
        self.components = list(components)
        graph_order = [info.name for info in bundle.graph.components]
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
    ) -> "TorchModelGraph":
        """Load a model bundle (folder or zip). See 'inference_model_core.load_graph' for the arguments.

        'device' is 'CPU', 'CUDA' or a torch device name for the components written in torch. Device names of
        another inference engine (e.g. OpenVINO's 'GPU') only apply to that engine: torch then uses the CPU.
        'num_threads' also sets torch's global intra-op threads.
        """
        options = LoadOptions(device=device, num_threads=num_threads, engine_options=engine_options or {})
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        bundle = ModelBundle.from_path(path_bundle)
        with bundle.open_dir() as path_dir:
            graph = bundle.graph
            infos = graph.components_for(list(outputs)) if outputs is not None else graph.components
            classes = {info.name: _component_class(info, inference_engine) for info in infos}
            engines = {component_class.engine for component_class in classes.values()}
            if inference_engine is not None and inference_engine not in engines:
                raise ValueError(
                    f"No component of '{path_bundle}' runs on inference engine '{inference_engine}'. Component "
                    f"types: {sorted({info.type for info in infos})}."
                )
            unknown_engines = sorted(set(options.engine_options) - engines)
            if unknown_engines:
                raise ValueError(f"'engine_options' are given for engines {unknown_engines}, which run no component.")
            param_overrides = _split_param_overrides(infos, classes, params or {})
            components = [
                classes[info.name].load(info, bundle, path_dir, options, params=param_overrides[info.name])
                for info in infos
            ]
        return cls(bundle, components)

    def write(self, path_bundle: PathLike, overwrite: bool = False) -> Path:
        """Write the model as a bundle. A path ending with '.zip' creates a zip, otherwise a folder."""
        loaded = {component.info.name for component in self.components}
        not_loaded = [info.name for info in self.bundle.graph.components if info.name not in loaded]
        if not_loaded:
            raise ValueError(f"Can't write a partially loaded graph. Components {not_loaded} are not loaded.")
        files: Dict[str, FileSource] = {}
        for component in self.components:
            files.update(component.files())
        write_by_suffix(self.bundle, path_bundle, files=files, overwrite=overwrite)
        return Path(path_bundle)

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
                Timing waits for a GPU to finish each component, which makes the run slightly slower.

        Returns:
            Output key -> one value per sample. Tensors are on the device of the component that produced them.
        """
        graph = self.bundle.graph
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
        with torch.no_grad():
            for name in needed:
                component = loaded[name]
                start = time.perf_counter()
                data.update(_run_component(component, data, batch_size))
                if timings is not None:
                    if component.device.type == "cuda":
                        torch.cuda.synchronize(component.device)  # GPU work is queued: wait for it to be timed
                    timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
        return {key: data[key] for key in output_keys}


def _component_class(info: ComponentInfo, inference_engine: Optional[str]) -> Type[Component]:
    if inference_engine is not None:
        component_class = find_alternative_class(info.type, inference_engine)
        if component_class is not None:
            return component_class
    return get_component_class(info.type)


def _split_param_overrides(
    infos: Sequence[ComponentInfo], classes: Mapping[str, Type[Component]], params: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Parameter overrides per component, from 'param' (all components with it) and 'component.param' keys."""
    overrides: Dict[str, Dict[str, Any]] = {info.name: {} for info in infos}
    parameters = {name: set(component_class.definition.model_fields) for name, component_class in classes.items()}
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
    device = component.device
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
            inputs = {port: _on_device(data[key][i], device) for port, key in info.inputs.items()}
            collect(component.forward(**inputs), n_samples=None)
        return results

    step = component.max_batch_size or max(batch_size, 1)
    for start in range(0, batch_size, step):
        stop = min(start + step, batch_size)
        inputs = {port: _stack(info, port, data[key][start:stop]).to(device) for port, key in info.inputs.items()}
        collect(component.forward(**inputs), n_samples=stop - start)
    return results


def _on_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _stack(info: ComponentInfo, port: str, values: Sequence[Any]) -> torch.Tensor:
    not_tensors = sorted({type(value).__name__ for value in values if not isinstance(value, torch.Tensor)})
    if not_tensors:
        raise TypeError(
            f"Can't batch input '{port}' of component '{info.name}': the samples are {not_tensors}. Batched "
            "components take tensors."
        )
    try:
        return torch.stack(list(values))
    except RuntimeError as e:
        shapes = sorted({tuple(value.shape) for value in values})
        raise ValueError(
            f"Can't batch input '{port}' of component '{info.name}': the samples have shapes {shapes}. Batched "
            "components need equally shaped values."
        ) from e
