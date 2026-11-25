"""The torch graph engine: loads the components of a bundle and runs them on a data dictionary.

Implements 'hafnia_inference_model.core.ModelGraph'. Values are kept per sample, without a batch axis: a 'batched'
component gets the tensors of all samples stacked on a new first axis and its outputs are split again, any other
component is called once per sample. Tensors are 'torch.Tensor's and stay on the device of the component that
produced them: the graph moves them only when the next component needs them elsewhere, so a graph of torch
components runs on the GPU from the image to the detections.
"""

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Unpack

import torch

from hafnia_inference_model.core.graph import LoadSettings
from hafnia_inference_model.core.model_bundle import ComponentInfo, ModelBundle
from hafnia_inference_model.torch.component import (
    ENGINE,
    Component,
    LoadOptions,
    find_alternative_class,
    get_component_class,
    is_torch_device,
)


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

    # --- loading -------------------------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        bundle: ModelBundle,
        outputs: Optional[Sequence[str]] = None,
        **load_settings: Unpack[LoadSettings],
    ) -> "TorchModelGraph":
        """Load a model bundle that has its files on disk. Called by 'hafnia_inference_model.core.load_graph', which
        describes the arguments.

        'device' is 'CPU', 'CUDA' or a torch device name for the components written in torch. Device names of
        another inference engine (e.g. OpenVINO's 'GPU') only apply to that engine: torch then uses the CPU. Such a
        name is an error when every component runs on torch, as no engine would use it.
        'num_threads' also sets torch's global intra-op threads.
        """
        inference_engine = load_settings.get("inference_engine")
        device = load_settings.get("device") or "CPU"
        num_threads = load_settings.get("num_threads")
        engine_options = load_settings.get("engine_options") or {}
        options = LoadOptions(device=device, num_threads=num_threads, engine_options=engine_options)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        with bundle.open_dir() as path_dir:
            graph = bundle.graph
            infos = graph.components_for(list(outputs)) if outputs is not None else graph.components
            classes = {info.name: _component_class(info, inference_engine) for info in infos}
            engines = {component_class.engine for component_class in classes.values()}
            if inference_engine is not None and inference_engine not in engines:
                raise ValueError(
                    f"No component of '{bundle.path}' runs on inference engine '{inference_engine}'. Component "
                    f"types: {sorted({info.type for info in infos})}."
                )
            if not is_torch_device(device) and engines <= {ENGINE}:
                raise ValueError(
                    f"Unsupported device '{device}'. The components of '{bundle.path}' run on torch, which takes "
                    "'CPU', 'CUDA' or a torch device name such as 'cuda:1'."
                )
            unknown_engines = sorted(set(options.engine_options) - engines)
            if unknown_engines:
                raise ValueError(f"'engine_options' are given for engines {unknown_engines}, which run no component.")
            param_overrides = _split_param_overrides(infos, classes, load_settings.get("params") or {})
            components = [
                classes[info.name].load(info, bundle, path_dir, options, params=param_overrides[info.name])
                for info in infos
            ]
        return cls(bundle, components)

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
