"""The inference model for hafnia: any model bundle behind hafnia's 'InferenceModel' interface.

'GraphInferenceModel' runs the bundle's model graph with the image as graph input 'image' and returns the graph
output 'predictions', so every bundle works with 'hafnia.dataset.benchmark'. There is one class for all model
families, engines and formats: what a model does is described by the components of its graph, and a graph
engine runs them.
"""

import time
from typing import Dict, List, Optional, Sequence, Unpack

import numpy as np
from hafnia.dataset.benchmark.inference_model import ImageType, InferenceModel
from hafnia.dataset.primitives import Primitive

from inference_model_core.bundle_io import PathLike
from inference_model_core.graph import LoadSettings, ModelGraph, load_graph
from inference_model_core.model_bundle import ModelBundle

INPUT_KEY = "image"
OUTPUT_KEY = "predictions"

LATENCY_TOTAL = "total"  # The latency stage of the whole 'predict_batch' call
LATENCY_PERCENTILES = (50, 90, 99)


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

    def measure_latency(self, images: Sequence[ImageType], batch_size: int = 1, warmup: int = 10) -> Dict[str, float]:
        """Time the components of 'predict_batch' over 'images', in batches of 'batch_size'.

        Measures synchronous latency: one call at a time, as 'predict_batch' runs. Throughput modes that overlap
        several requests (e.g. OpenVINO async infer requests) give different numbers and are not covered.

        Images should be loaded arrays, so that reading files is not measured. The first 'warmup' batches
        (cycling through the images) are run but not timed: engines allocate memory and pick kernels on the
        first calls.

        Returns per stage the mean and the 'LATENCY_PERCENTILES' latency per batch in milliseconds (e.g.
        'network_p90_ms'), and 'throughput_images_per_s' over the timed batches. The stages are the components
        that compute the predictions, in graph order (e.g. 'resize', 'network', 'nms'), and 'total', which also
        covers the time between the components (batching the values). A component is charged with moving its
        inputs to its device.
        """
        if batch_size < 1:
            raise ValueError(f"'batch_size' must be at least 1, got {batch_size}.")
        if len(images) == 0:
            raise ValueError("At least one image is needed.")
        batches = [list(images[start : start + batch_size]) for start in range(0, len(images), batch_size)]

        for i_warmup in range(warmup):
            self.predict_batch(batches[i_warmup % len(batches)])

        components = self.get_model_info().graph.components_for([OUTPUT_KEY])
        stages = [*(component.name for component in components), LATENCY_TOTAL]
        timings: Dict[str, List[float]] = {stage: [] for stage in stages}
        n_images = 0
        for batch in batches:
            batch_timings: Dict[str, float] = {}
            start = time.perf_counter()
            self.graph.run_batch({INPUT_KEY: batch}, outputs=[OUTPUT_KEY], timings=batch_timings)
            batch_timings[LATENCY_TOTAL] = time.perf_counter() - start
            for stage in stages:
                timings[stage].append(batch_timings[stage])
            n_images += len(batch)

        report = {}
        for stage, seconds in timings.items():
            milliseconds = np.asarray(seconds) * 1000
            report[f"{stage}_mean_ms"] = float(milliseconds.mean())
            for percentile in LATENCY_PERCENTILES:
                report[f"{stage}_p{percentile}_ms"] = float(np.percentile(milliseconds, percentile))
        report["throughput_images_per_s"] = n_images / sum(timings[LATENCY_TOTAL])
        return report
