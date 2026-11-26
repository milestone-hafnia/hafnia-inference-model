"""The model graph: the components of a bundle, loaded by the torch graph engine to run on a data dictionary.

All inputs, outputs and intermediate values are items of one dictionary. Each component reads the keys its
inputs are connected to and writes the keys of its outputs. Values are kept per sample, without a batch axis:
a 'batched' component gets the values of all samples stacked on a new first axis and its outputs are split
again, any other component is called once per sample.

The torch graph engine ('hafnia_model_bundle.torch') has an implementation of each component definition and
exchanges 'torch.Tensor's between them. It is the only graph engine and is imported when a graph is loaded, so
describing, reading and writing a bundle needs no torch.
"""

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, TypedDict, Union, Unpack

from hafnia_model_bundle.core.bundle_io import PathLike
from hafnia_model_bundle.core.model_bundle import ModelBundle


class LoadSettings(TypedDict, total=False):
    """How a bundle is loaded: the keyword arguments of 'GraphInferenceModel.from_path', 'from_bundle' and
    'load_graph'. They only apply to the loaded model and are not part of the model info. A setting that is left
    out or 'None' has its default.

    Keys:
        inference_engine: Inference engine for every component that has an implementation on it, e.g. 'openvino'
            runs the ONNX networks of a bundle on OpenVINO. By default each component runs on its default
            implementation.
        params: Overrides of component parameters. '{"score_threshold": 0.5}' applies to every component with
            that parameter, '{"nms.score_threshold": 0.5}' to component 'nms'.
        device: Device for the inference engines. Defaults to 'CPU'.
        num_threads: Threads per inference engine. By default the engines decide.
        engine_options: Inference engine name -> settings only that engine knows, e.g.
            '{"openvino": {"performance_hint": "LATENCY"}}'.
    """

    inference_engine: Optional[str]
    params: Optional[Mapping[str, Any]]
    device: Optional[str]
    num_threads: Optional[int]
    engine_options: Optional[Mapping[str, Mapping[str, Any]]]


class ModelGraph(Protocol):
    """A loaded model graph: what this package uses of 'hafnia_model_bundle.torch.TorchModelGraph', which it can
    not import here without importing torch."""

    bundle: ModelBundle

    @property
    def components(self) -> Sequence[Any]:
        """The loaded component implementations, in graph order."""
        ...

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


def load_graph(
    bundle: Union[ModelBundle, PathLike],
    outputs: Optional[Sequence[str]] = None,
    **load_settings: Unpack[LoadSettings],
) -> ModelGraph:
    """Load the model graph of a bundle.

    Args:
        bundle: A model bundle that was read or written, or the path of a bundle folder or zip.
        outputs: Only load the components needed to compute these data keys. By default all components.
        **load_settings: How the bundle is loaded, e.g. 'device="CUDA"'. See 'LoadSettings'.
    """
    unknown = sorted(set(load_settings) - set(LoadSettings.__annotations__))
    if unknown:
        raise TypeError(f"Unknown load settings {unknown}. Settings: {sorted(LoadSettings.__annotations__)}")
    try:
        # Imported here: only running a graph needs torch, the bundle definition does not
        from hafnia_model_bundle.torch.graph import TorchModelGraph
    except ImportError as e:
        raise ImportError(
            "Running a model graph needs the torch graph engine ('hafnia_model_bundle.torch'), which can not be "
            f"imported: {e}. Install the 'torch' extra."
        ) from e
    if not isinstance(bundle, ModelBundle):
        bundle = ModelBundle.from_path(bundle)
    return TorchModelGraph.load(bundle, outputs=outputs, **load_settings)
