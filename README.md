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
| `inference_model_openvino` | OpenVINO flavors (`OpenVINOTorchvisionSSD`, `OpenVINOTorchvisionDetection`) for ONNX and IR bundles, ONNX → OpenVINO conversion, INT8 quantization (NNCF) | Implemented |
| `model_exporters` | Framework → ONNX bundle exporters (torchvision SSD and Faster R-CNN) | Implemented |
| `package_openvino` | Trainer glue: checkpoint and dataset loading, latency measurement, class mappings | Implemented |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/export_torchvision_to_onnx.py` | Export a pretrained torchvision detection model to an ONNX bundle in the experiment model folder |
| `scripts/export_onnx_to_openvino.py` | Convert an ONNX bundle (checkpoint or `--model-path`) to an OpenVINO IR bundle: FP32, FP16 or INT8 (calibrated on a Hafnia dataset) |
| `scripts/benchmark.py` | Accuracy (Hafnia metrics, optional class mappings) and latency per stage (preprocess, model, postprocess) of any bundle on any engine and device |

## Examples

| Example | Shows |
|---|---|
| `examples/example_inference_model.py` | The basic flow: torchvision → ONNX bundle, run it with ONNX Runtime, convert to OpenVINO, run it with OpenVINO (same loading and `predict` code for both engines), draw the predictions next to the ground truth, and run Hafnia's benchmark |
| `examples/example_compare_models.py` | SSDLite320 and Faster R-CNN on ONNX Runtime and OpenVINO (ONNX, IR FP32, IR FP16, IR INT8) in one loop, with Hafnia's benchmark (mAP) and latency in a table |

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

## Trainer package

```bash
hafnia trainer create-zip .           # -> trainer.zip (see .hafniaignore)
hafnia runc build-local trainer.zip   # -> Docker image, its tag in state.json
```

`hafnia runc launch-local` mounts only the dataset and needs the NVIDIA container runtime. To test a script
that takes an input bundle, run the image like the platform does, with the checkpoint, model and output
folders mounted (here the bundle from `.data/runs/1/model` is the checkpoint of run 2):

```bash
IMAGE=$(python -c "import json; print(json.load(open('state.json'))['image_tag'])")
mkdir -p .data/runs/2/checkpoints && cp .data/runs/1/model/*.zip .data/runs/2/checkpoints/
docker run --rm -e HAFNIA_CLOUD=true -e PYTHONPATH=src \
  -v "$PWD/tests/data/micro-coco-2017:/opt/ml/input/data/training:ro" \
  -v "$PWD/.data/runs/2/checkpoints:/opt/ml/checkpoints" \
  -v "$PWD/.data/runs/2/model:/opt/ml/model" \
  -v "$PWD/.data/runs/2/output:/opt/ml/output/data" \
  "$IMAGE" python scripts/export_onnx_to_openvino.py --precision int8 --calibration-split-name all
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

# INT8 quantization, calibrated on a Hafnia dataset ('name:version' or a dataset folder)
python scripts/export_onnx_to_openvino.py --model-path ssdlite320_mobilenet_v3_large_onnx.zip --precision int8 \
    --calibration-dataset coco-2017:1.0.0

# Benchmark accuracy and latency of any bundle (here an ONNX bundle run on OpenVINO)
python scripts/benchmark.py --model-path ssdlite320_mobilenet_v3_large_onnx.zip --inference-engine openvino
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
