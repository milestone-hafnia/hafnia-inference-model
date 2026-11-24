"""The model graph: the components of a bundle, loaded by a graph engine to run on a data dictionary.

All inputs, outputs and intermediate values are items of one dictionary. Each component reads the keys its
inputs are connected to and writes the keys of its outputs. Values are kept per sample, without a batch axis:
a 'batched' component gets the values of all samples stacked on a new first axis and its outputs are split
again, any other component is called once per sample.

A graph engine has an implementation of each component definition and decides how tensors are exchanged between
them. This repo has one: 'torch' ('inference_model_torch'), which exchanges 'torch.Tensor's. Graph engines are
listed in the 'hafnia.inference_model_graph_engines' entry point group and imported only when used.
"""

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from inference_model_core.bundle import PathLike
from inference_model_core.model_info import ModelBundleInfo

ENTRY_POINT_GROUP = "hafnia.inference_model_graph_engines"
DEFAULT_GRAPH_ENGINE = "torch"


class ModelGraph(Protocol):
    """A loaded model graph, as every graph engine provides it."""

    model_info: ModelBundleInfo
    components: Sequence[Any]  # The loaded component implementations of the graph engine, in graph order

    def run(self, inputs: Mapping[str, Any], outputs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Run the graph on one sample: data dictionary with the graph inputs in, the graph outputs out."""
        ...

    def run_batch(
        self,
        inputs: Mapping[str, Sequence[Any]],
        outputs: Optional[Sequence[str]] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[Any]]:
        """Run the graph on a batch of samples: graph input key -> one value per sample in, output key -> one
        value per sample out. 'outputs' can name intermediate data keys: only the components needed for them are
        run. 'timings' collects the seconds spent per component."""
        ...

    def write(self, path_bundle: PathLike, overwrite: bool = False) -> Path:
        """Write the model as a bundle."""
        ...


def list_graph_engines() -> List[str]:
    return sorted(entry_point.name for entry_point in entry_points(group=ENTRY_POINT_GROUP))


def load_graph(
    path_bundle: PathLike,
    graph_engine: str = DEFAULT_GRAPH_ENGINE,
    inference_engine: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    device: str = "CPU",
    num_threads: Optional[int] = None,
    engine_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    outputs: Optional[Sequence[str]] = None,
) -> ModelGraph:
    """Load the model graph of a bundle (zip or directory).

    Args:
        path_bundle: Zip or directory bundle.
        graph_engine: Graph engine that runs the components.
        inference_engine: Inference engine for every component that has an implementation on it, e.g. 'openvino'
            runs the ONNX networks of a bundle on OpenVINO. Other components run on their default implementation.
        params: Overrides of component parameters. '{"score_threshold": 0.5}' applies to every component
            with that parameter, '{"nms.score_threshold": 0.5}' to component 'nms'. Overrides only apply
            to this instance, not to written bundles.
        device: Device for the inference engines, e.g. 'CPU'.
        num_threads: Threads per inference engine. 'None' uses the engine defaults.
        engine_options: Inference engine name -> settings only that engine knows, e.g.
            '{"openvino": {"performance_hint": "LATENCY"}}'.
        outputs: Only load the components needed to compute these data keys. By default all components.
    """
    matches = [entry_point for entry_point in entry_points(group=ENTRY_POINT_GROUP) if entry_point.name == graph_engine]
    if not matches:
        raise KeyError(f"No graph engine '{graph_engine}'. Available: {list_graph_engines()}")
    try:
        graph_class = matches[0].load()
    except ImportError as e:
        raise ImportError(
            f"Graph engine '{graph_engine}' ('{matches[0].value}') can not be imported: {e}. Install its dependencies."
        ) from e
    return graph_class.load(
        path_bundle,
        inference_engine=inference_engine,
        params=params,
        device=device,
        num_threads=num_threads,
        engine_options=engine_options,
        outputs=outputs,
    )
