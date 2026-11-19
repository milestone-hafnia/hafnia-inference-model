# package-openvino

A Hafnia trainer package for **generic model export and inference**. It does no training. Instead it
converts models to OpenVINO and runs them through one shared `InferenceModel` interface that always
returns Hafnia primitives, so models can be visualized, benchmarked and served the same way.

See [docs/design.md](docs/design.md) for the concept and decisions.

## Layout

Each top-level package under `src/` is a separate concept that can later move to its own repo:

| Package | Purpose | Status |
|---|---|---|
| `inference_model_core` | Model info schema, model bundles (zip/dir), `InferenceModelBase`, flavor registry, `inference_model_from_path`, shared preprocessing, NumPy NMS and primitive helpers. Engine agnostic, torch free. | Implemented |
| `inference_model_onnxruntime` | ONNX Runtime flavors: `OnnxRuntimeTorchvisionSSD`, `OnnxRuntimeTorchvisionDetection` | Implemented |
| `inference_model_openvino` | OpenVINO flavors (`OpenVINOTorchvisionSSD`, `OpenVINOTorchvisionDetection`) for ONNX and IR bundles, ONNX → OpenVINO conversion. NNCF quantization is planned. | Implemented |
| `model_exporters` | Framework → ONNX bundle exporters (torchvision SSD and Faster R-CNN) | Implemented |
| `package_openvino` | Trainer glue: checkpoint discovery, later benchmarking | Started |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/export_torchvision_to_onnx.py` | Export a pretrained torchvision detection model to an ONNX bundle in the experiment model folder |
| `scripts/export_onnx_to_openvino.py` | Convert an ONNX bundle (checkpoint or `--model-path`) to an OpenVINO IR bundle, FP32 or FP16 |

## Examples

| Example | Shows |
|---|---|
| `examples/example_inference_model.py` | The basic flow: torchvision → ONNX bundle, run it with ONNX Runtime, convert to OpenVINO, run it with OpenVINO (same loading and `predict` code for both engines), draw the predictions next to the ground truth, and run Hafnia's benchmark |
| `examples/example_compare_models.py` | SSDLite320 and Faster R-CNN on ONNX Runtime and OpenVINO (ONNX, IR FP32, IR FP16) in one loop, with Hafnia's benchmark (mAP) and latency in a table, and predictions drawn per model |

```bash
uv run python examples/example_inference_model.py  # Outputs go to .data/examples/
uv run python examples/example_compare_models.py
```

## Micro datasets

Tests and examples use small copies of Hafnia platform datasets stored in `tests/data`, so they run offline.
`tests/micro_datasets.py` defines them (source dataset, number of samples, seed) and creates them from the
platform. Creating them needs a Hafnia profile with a valid API key; tests create a missing micro dataset
automatically.

| Micro dataset | Source | Samples |
|---|---|---|
| `micro-coco-2017` | `coco-2017-tiny` 1.0.0 (20 samples) | 8, seed 0 |

```bash
uv run python tests/micro_datasets.py          # Create missing micro datasets
uv run python tests/micro_datasets.py --force  # Recreate all of them
```

## Development

```bash
uv sync --all-extras         # Install dependencies incl. torch, onnxruntime, openvino and the dev group
uv run pytest                # Run all tests (the first run downloads pretrained torchvision weights)
uv run pytest -m "not slow"  # Skip the end-to-end export tests
uv run pytest tests/inference_model_openvino  # Tests of one module (one folder per module under src/)
uv run ruff check src tests scripts examples  # Lint
uv run ruff format src tests scripts examples # Format
uv run mypy src tests scripts examples        # Type check
```

## Usage

```bash
# Export SSDLite320 (default) or Faster R-CNN to an ONNX bundle
python scripts/export_torchvision_to_onnx.py
python scripts/export_torchvision_to_onnx.py --model-name fasterrcnn_mobilenet_v3_large_320_fpn

# Convert an ONNX bundle to OpenVINO IR (FP32 by default, or --precision fp16)
python scripts/export_onnx_to_openvino.py --model-path ssdlite320_mobilenet_v3_large_onnx.zip
```

```python
from inference_model_core import inference_model_from_path

model = inference_model_from_path("model.zip")                       # Flavor is selected from model_info.json
model = inference_model_from_path(
    "model.zip",
    inference_engine="openvino",                     # Optional: override the bundle's default engine
    postprocessing_params={"score_threshold": 0.3},  # Optional: override decoder defaults
    device="CPU",                                    # Engine option
)
primitives = model.predict(image)                    # List[Primitive], e.g. Bbox predictions
```
