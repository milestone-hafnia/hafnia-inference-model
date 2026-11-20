# package-openvino

A Hafnia trainer package for **generic model export and inference**. It does no training. Instead it
converts models to OpenVINO and runs them through one shared `InferenceModel` interface that always
returns Hafnia primitives, so models can be visualized, benchmarked and served the same way.

A model is stored as a **model bundle**: a zip with the model files and a `model_info.json` that describes
the model as a **graph of components**, for example

```
image -> load_image -> resize -> to_tensor -> network -> nms -> to_bboxes -> predictions
          (numpy)     (numpy)    (numpy)     (onnx)   (numpy)   (numpy)
```

Each component reads and writes named items of one data dictionary and has a type, a runtime (NumPy, torch,
ONNX, OpenVINO), parameters and the conversions it supports. One executor runs any graph, the engines only
run `network` components, and converting a bundle (torch → ONNX → OpenVINO → INT8) replaces the network and
keeps the rest. A new model family is a new graph built from existing components.

See [docs/design.md](docs/design.md) for the concept and decisions.

## Supported models

| Model | Task | Sources | Network | After the network (NumPy components) |
|---|---|---|---|---|
| torchvision SSDLite320 MobileNetV3 | detection | pretrained | backbone, head, box decoding | per-class threshold, top-k and NMS |
| torchvision Faster R-CNN MobileNetV3 320 FPN | detection | pretrained | the full model incl. resize and NMS | score threshold |
| RF-DETR Nano / Small / Medium / Large | detection | pretrained, trainer-object-detection archive | RF-DETR's own export | sigmoid and top-k over queries and classes |
| torchvision ResNet18 | classification | pretrained, trainer-classification archive | the model | softmax, top-1 |

Each network runs on PyTorch (the source bundle), ONNX Runtime and OpenVINO (ONNX or IR in FP32, FP16 or INT8).

## Layout

Each top-level package under `src/` is a separate concept that can later move to its own repo:

| Package | Purpose |
|---|---|
| `inference_model_core` | Model info schema, model bundles (zip/dir), components and their registry, `ModelGraph` (the executor), `GraphInferenceModel` and `inference_model_from_path`, `convert_bundle`, and the standard NumPy components (image loading, resize, tensor conversion, NMS, query top-k, primitives). Engine agnostic, torch free. |
| `inference_model_onnxruntime` | Runs ONNX `network` components on ONNX Runtime |
| `inference_model_openvino` | Runs ONNX and OpenVINO IR `network` components on OpenVINO. Converters ONNX → OpenVINO IR (FP32, FP16) and INT8 quantization (NNCF) |
| `inference_model_torch` | Base class of torch networks and the converter torch → ONNX |
| `model_exporters` | Model families: the torch network components of torchvision SSD, Faster R-CNN and ResNet18 and of RF-DETR, and the functions that write their bundles from pretrained models and trainer archives |
| `package_openvino` | Trainer glue: checkpoint and dataset loading, latency per component, metrics, class mappings |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/export_to_onnx.py` | Export a trainer archive (checkpoint or `--model-path`) or a pretrained model (`--model-name`) to an ONNX bundle in the experiment model folder |
| `scripts/export_onnx_to_openvino.py` | Convert an ONNX bundle (checkpoint or `--model-path`) to an OpenVINO IR bundle: FP32, FP16 or INT8 (calibrated on a Hafnia dataset) |
| `scripts/benchmark.py` | Accuracy (Hafnia metrics, optional class mappings) and latency per component (e.g. resize, network, NMS) of any bundle on any engine and device |

## Examples

| Example | Shows |
|---|---|
| `examples/example_inference_model.py` | The basic flow: torchvision → model bundle → ONNX bundle, run it with ONNX Runtime, convert to OpenVINO, run it with OpenVINO (same loading and `predict` code for both engines), draw the predictions, and run Hafnia's benchmark |
| `examples/example_compare_models.py` | SSDLite320, Faster R-CNN and RF-DETR Nano on PyTorch, ONNX Runtime and OpenVINO (ONNX, IR FP32, IR FP16, IR INT8) in one loop, with Hafnia's benchmark (mAP) and latency in a table |

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
uv sync --all-extras         # Install dependencies incl. torch, rfdetr, onnxruntime, openvino and the dev group
uv run pytest                # Run all tests (the first run downloads pretrained torchvision weights)
uv run pytest -m "not slow"  # Skip the end-to-end export tests
uv run pytest tests/inference_model_openvino  # Tests of one module (one folder per module under src/)
uv run ruff check src tests scripts examples  # Lint
uv run ruff format src tests scripts examples # Format
uv run mypy src tests scripts examples        # Type check
```

## Usage

```bash
# Export a pretrained model to an ONNX bundle (default: SSDLite320)
python scripts/export_to_onnx.py
python scripts/export_to_onnx.py --model-name RFDETRNano

# Export a trainer-classification or trainer-object-detection model archive (e.g. a platform checkpoint)
python scripts/export_to_onnx.py --model-path ../trainer-classification/pretrained_models/resnet18.zip

# Convert an ONNX bundle to OpenVINO IR (FP32 by default, or --precision fp16)
python scripts/export_onnx_to_openvino.py --model-path ssdlite320_mobilenet_v3_large_onnx.zip

# INT8 quantization, calibrated on a Hafnia dataset ('name:version' or a dataset folder)
python scripts/export_onnx_to_openvino.py --model-path ssdlite320_mobilenet_v3_large_onnx.zip --precision int8 \
    --calibration-dataset coco-2017:1.0.0

# Benchmark accuracy and latency of any bundle (here an ONNX bundle run on OpenVINO)
python scripts/benchmark.py --model-path ssdlite320_mobilenet_v3_large_onnx.zip --inference-engine openvino
```

```python
from inference_model_core import convert_bundle, inference_model_from_path, read_model_info
from model_exporters.registry import create_pretrained_bundle

# Write a pretrained model as a bundle with a torch network, and convert it. Only this part needs torch.
create_pretrained_bundle("ssdlite320_mobilenet_v3_large", "model_torch.zip")
convert_bundle("model_torch.zip", "model_onnx.zip", target="onnx")
convert_bundle("model_onnx.zip", "model_openvino.zip", target="openvino")  # Or options={"precision": "fp16"}
convert_bundle(
    "model_onnx.zip",
    "model_int8.zip",
    target="openvino_int8",
    calibration_inputs=[{"image": path} for path in calibration_image_paths],
)

# Run any bundle: the graph in model_info.json says how
model = inference_model_from_path("model_onnx.zip")
model = inference_model_from_path(
    "model_onnx.zip",
    inference_engine="openvino",  # Optional: run the ONNX network on OpenVINO
    params={"score_threshold": 0.3},  # Optional: override component parameters ("nms.score_threshold" for one)
    device="CPU",  # Engine option
)
primitives = model.predict(image)  # List[Primitive], e.g. Bbox predictions

# The graph and intermediate values
print([f"{c.name} ({c.type}, {c.runtime})" for c in read_model_info("model_onnx.zip").graph.components])
network_outputs = model.graph.run({"image": image}, outputs=["anchor_boxes", "anchor_scores"])
```
