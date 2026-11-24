"""The inference model for hafnia: any model bundle behind hafnia's 'InferenceModel' interface.

'GraphInferenceModel' runs the bundle's model graph with the image as graph input 'image' and returns the graph
output 'predictions', so every bundle works with 'hafnia.dataset.benchmark'. There is one class for all model
families, engines and formats: what a model does is described by the components of its graph, and a graph
engine runs them.
"""

from typing import List, Optional, Unpack

from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.primitives import Primitive

from inference_model_core.bundle_io import PathLike
from inference_model_core.graph import LoadSettings, ModelGraph, load_graph
from inference_model_core.model_bundle import ModelBundle

INPUT_KEY = "image"
OUTPUT_KEY = "predictions"


class GraphInferenceModel(InferenceModel):
    def __init__(self, graph: ModelGraph):
        graph_info = graph.bundle.graph
        if graph_info.inputs != [INPUT_KEY] or OUTPUT_KEY not in graph_info.outputs:
            raise ValueError(
                f"An inference model needs a graph with the input '{INPUT_KEY}' and an output '{OUTPUT_KEY}', got "
                f"inputs {graph_info.inputs} and outputs {graph_info.outputs}. Use 'load_graph' for other graphs."
            )
        self.graph = graph

    @classmethod
    def from_path(cls, path_bundle: PathLike, **load_settings: Unpack[LoadSettings]) -> "GraphInferenceModel":
        """Load the model bundle in a folder or zip as an inference model. See 'from_bundle' for the arguments."""
        return cls.from_bundle(ModelBundle.from_path(path_bundle), **load_settings)

    @classmethod
    def from_bundle(cls, bundle: ModelBundle, **load_settings: Unpack[LoadSettings]) -> "GraphInferenceModel":
        """Load a model bundle as an inference model.

        Args:
            bundle: A bundle that was read or written, e.g. the result of 'ModelBundle.convert'. Its files are
                read from 'bundle.path', so a bundle that is only in memory has to be written first.
            **load_settings: How the bundle is loaded: 'inference_engine', 'params', 'device', 'num_threads' and
                'engine_options', e.g. 'params={"score_threshold": 0.5}, device="CUDA"'. See 'LoadSettings' for
                each of them.
        """
        return cls(load_graph(bundle, **load_settings))

    def predict(self, image: ImageType, sample_dict: Optional[dict] = None) -> List[Primitive]:
        return self.graph.run({INPUT_KEY: image}, outputs=[OUTPUT_KEY])[OUTPUT_KEY]

    def predict_batch(
        self,
        images: List[ImageType],
        sample_dicts: Optional[List[dict]] = None,
    ) -> List[List[Primitive]]:
        if sample_dicts is not None and len(sample_dicts) != len(images):
            raise ValueError(
                f"Length of 'sample_dicts' ({len(sample_dicts)}) must match length of 'images' ({len(images)})."
            )
        return self.graph.run_batch({INPUT_KEY: images}, outputs=[OUTPUT_KEY])[OUTPUT_KEY]

    def get_model_info(self) -> ModelBundle:
        return self.graph.bundle
