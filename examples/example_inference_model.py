"""Generic model export and inference:

    1) torchvision model -> model bundle -> ONNX bundle
    2) Run the ONNX bundle with ONNX Runtime
    3) Convert the ONNX bundle to an OpenVINO bundle
    4) Run the OpenVINO bundle with OpenVINO
    5) Visualize the predictions
    6) Benchmark the model on a dataset with Hafnia's benchmark

A model bundle is a zip with the model files and a 'model_info.json' that describes the model: its tasks and
a graph of components (e.g. resize -> to_tensor -> network -> nms -> to_bboxes) with their parameters.
'GraphInferenceModel.from_bundle' (or 'from_path') runs that graph, whatever the model family or the engine of
the network, so steps 2 and 4 are the same code. Converting a bundle replaces the network component and keeps
the rest.
Predictions are Hafnia primitives, here 'Bbox' with normalized coordinates.

Run from the repo root (needs all extras: 'uv sync --all-extras'):

    uv run python examples/example_inference_model.py

See 'example_compare_models.py' for several models and engines benchmarked side by side.
"""

import time
from pathlib import Path

from hafnia.dataset.benchmark.benchmark import InferenceModel, run_benchmark
from hafnia.dataset.hafnia_dataset import HafniaDataset, Primitive
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.image_visualizations import draw_annotations
from PIL import Image
from rich import print as rprint

from hafnia_inference_model import GraphInferenceModel, ModelBundle
from hafnia_inference_model.core import AlignPredictionClasses

t0 = time.time()
##### PREPARE EXAMPLE #####


PATH_OUTPUT = Path(".data") / "examples" / "inference_model"

#  An image from the micro dataset (8 COCO images, see 'tests/micro_datasets.py')
path_dataset = Path(__file__).parents[1] / "tests" / "data" / "micro-coco-2017"
dataset = HafniaDataset.from_path(path_dataset)
sample = Sample(**dataset[4])
image = sample.read_image()

# To align coco pretrained models (91 classes) to our coco dataset that only contain the 80 classes
# that is actually being used in coco
align_class_names = [AlignPredictionClasses()]
##########################

# 1) Write a pretrained torchvision model as a model bundle (with a torch network) and convert it to an ONNX
# bundle. This is the only step that needs torch.
MODEL_NAME = "ssdlite320_mobilenet_v3_large"  # Or "fasterrcnn_mobilenet_v3_large_320_fpn"
bundle_torch = ModelBundle.from_pretrained(
    MODEL_NAME,
    path_bundle=PATH_OUTPUT / f"{MODEL_NAME}_torch.zip",
    overwrite=True,
)

path_onnx = PATH_OUTPUT / f"{MODEL_NAME}_onnx.zip"
bundle_onnx = bundle_torch.convert(target="onnx", path_bundle=path_onnx, overwrite=True)

rprint(f"1) Exported '{bundle_onnx.path}'")
rprint("   Graph: " + " -> ".join(f"{c.name} ({c.type})" for c in bundle_onnx.graph.components))

# 2) Load the ONNX bundle and run inference with ONNX Runtime
model_onnx: InferenceModel = GraphInferenceModel.from_bundle(bundle=bundle_onnx, params={"score_threshold": 0.6})
predictions_onnx: list[Primitive] = model_onnx.predict(image)

# 3) Convert the ONNX bundle to OpenVINO IR
path_openvino = PATH_OUTPUT / f"{MODEL_NAME}_openvino.zip"
bundle_openvino = bundle_onnx.convert(target="openvino", path_bundle=path_openvino, overwrite=True)
rprint(f"3) Converted to '{bundle_openvino.path}'")

# 4) Load the OpenVINO bundle and run inference with OpenVINO
model_openvino: InferenceModel = GraphInferenceModel.from_bundle(
    bundle=bundle_openvino, params={"score_threshold": 0.3}
)
predictions_openvino: list[Primitive] = model_openvino.predict(image)

# 5) Model predictions are standardized with Hafnia primitives
image_openvino = draw_annotations(image=image, primitives=predictions_openvino)
path_visualization = PATH_OUTPUT / "ground_truth_onnx_openvino.png"
Image.fromarray(image_openvino).save(path_visualization)
rprint(f"5) Visualization saved to '{path_visualization}'")

# 6) Model benchmarking: run the OpenVINO model on the micro COCO dataset
model_benchmark = GraphInferenceModel.from_path(path_openvino)
metrics, dataset_predictions = run_benchmark(dataset, model_benchmark, recipe_transforms=align_class_names)
rprint(f"6) Benchmark metrics: {metrics}")

t1 = time.time()
print(f"Total time: {t1 - t0:.2f}s")
