"""Benchmark several models on several engines and formats with the same code.

For each pretrained detection model, the example writes a model bundle with a torch network and converts it to
ONNX and to OpenVINO IR in FP32, FP16 and INT8 (NNCF post-training quantization). Every bundle is then loaded
with 'GraphInferenceModel.from_path' and run through Hafnia's benchmark (accuracy) and a simple latency measurement:

    model                 network            engine
    SSDLite320            torch              PyTorch
    Faster R-CNN 320      ONNX               ONNX Runtime
    RF-DETR Nano          ONNX               OpenVINO (engine override)
                          OpenVINO IR FP32   OpenVINO
                          OpenVINO IR FP16   OpenVINO
                          OpenVINO IR INT8   OpenVINO

INT8 is calibrated on the same 8 images it is evaluated on, which flatters it: use a larger, separate dataset
for real numbers (at least 300 calibration images are recommended).

The loop body never depends on the model or the engine: the model info in each bundle describes the model as a
graph of components, and a conversion only replaces the network component. See 'example_inference_model.py'
for the basic flow of a single model.

Run from the repo root (needs all extras: 'uv sync --all-extras'):

    uv run python examples/example_compare_models.py
"""

import contextlib
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.utils import progress_bar
from rich import print as rprint
from rich.table import Table

from inference_model_core import GraphInferenceModel, ModelBundle

MODELS = {
    "SSDLite320": "ssdlite320_mobilenet_v3_large",
    "Faster R-CNN 320": "fasterrcnn_mobilenet_v3_large_320_fpn",
    "RF-DETR Nano": "RFDETRNano",
}
SERVING_PARAMS = {"score_threshold": 0.3}  # Component parameters for latency and visualization
PATH_DATASET = Path(__file__).parents[1] / "tests" / "data" / "micro-coco-2017"
PATH_OUTPUT = Path(".data") / "examples" / "compare_models"
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

# (bundle path, engine). 'None' runs the bundle on its default engine.
BundleRun = Tuple[Path, Optional[str]]


def export_bundles(model_name: str, calibration_images: List[np.ndarray]) -> Dict[str, BundleRun]:
    """Write a pretrained model as a bundle, convert and quantize it, and list the ways to run the bundles."""
    path_torch = PATH_OUTPUT / f"{model_name}_torch.zip"
    path_onnx = PATH_OUTPUT / f"{model_name}_onnx.zip"
    path_fp32 = PATH_OUTPUT / f"{model_name}_openvino_fp32.zip"
    path_fp16 = PATH_OUTPUT / f"{model_name}_openvino_fp16.zip"
    path_int8 = PATH_OUTPUT / f"{model_name}_openvino_int8.zip"
    bundle_torch = ModelBundle.from_pretrained(model_name, path_torch, overwrite=True)
    bundle_onnx = bundle_torch.convert("onnx", path_onnx, overwrite=True)
    bundle_onnx.convert("openvino", path_fp32, overwrite=True)
    bundle_onnx.convert("openvino", path_fp16, options={"precision": "fp16"}, overwrite=True)
    calibration_inputs = [{"image": image} for image in calibration_images]
    bundle_onnx.convert("openvino_int8", path_int8, calibration_inputs=calibration_inputs, overwrite=True)
    return {
        "torch / PyTorch": (path_torch, None),
        "ONNX / ONNX Runtime": (path_onnx, None),
        "ONNX / OpenVINO": (path_onnx, "openvino"),
        "IR FP32 / OpenVINO": (path_fp32, None),
        "IR FP16 / OpenVINO": (path_fp16, None),
        "IR INT8 / OpenVINO": (path_int8, None),
    }


def median_latency_ms(path_bundle: Path, engine: Optional[str], images: List[np.ndarray], repeats: int = 5) -> float:
    """End-to-end latency of 'predict' (all components of the graph) for one image, on CPU."""
    model = GraphInferenceModel.from_path(path_bundle, inference_engine=engine, params=SERVING_PARAMS)
    model.predict(images[0])  # Warm-up
    latencies = []
    for image in images * repeats:
        start = time.perf_counter()
        model.predict(image)
        latencies.append(time.perf_counter() - start)
    return float(np.median(latencies)) * 1000


# A Hafnia dataset to run on: the micro dataset (8 COCO images, see 'tests/micro_datasets.py'). Any
# Hafnia dataset works, e.g. 'HafniaDataset.from_name("coco-2017")'.
dataset = HafniaDataset.from_path(PATH_DATASET)
images = [Sample(**sample_dict).read_image() for sample_dict in dataset]

table = Table(
    title=f"Detection models on {len(dataset)} COCO images",
    caption=f"ms: median latency per image, CPU, score threshold {SERVING_PARAMS['score_threshold']}",
)
for column in ("Model", "Network / engine", "mAP", "mAP@50", "ms"):
    table.add_column(column)

n_model_families = len(MODELS)
for i_model_family, (model_label, model_name) in enumerate(MODELS.items()):
    bundle_runs = export_bundles(model_name, calibration_images=images)
    for i_run, (run_label, (path_bundle, engine)) in progress_bar(
        enumerate(bundle_runs.items()),
        description=f"Benchmarking {model_label} ({i_model_family + 1}/{n_model_families})",
        total=len(bundle_runs),
    ):
        # Accuracy with the bundle's default (low) score threshold, as metrics need all detections
        model = GraphInferenceModel.from_path(path_bundle, inference_engine=engine)
        with contextlib.redirect_stdout(io.StringIO()):  # Hide the COCO evaluation report
            metrics, _ = model.benchmark(dataset)

        latency_ms = median_latency_ms(path_bundle, engine, images)
        mAP = metrics["object_detection/mAP"]
        mAP_50 = metrics["object_detection/mAP_50"]
        table.add_row(model_label if i_run == 0 else "", run_label, f"{mAP:.3f}", f"{mAP_50:.3f}", f"{latency_ms:.1f}")
    table.add_section()


rprint(table)
rprint(f"Bundles in '{PATH_OUTPUT}'")
