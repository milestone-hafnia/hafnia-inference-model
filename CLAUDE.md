# package-openvino

A Hafnia trainer package for generic model export and inference. It does no training: it converts model
bundles to OpenVINO and runs any bundle through one `InferenceModel` interface that returns Hafnia primitives.
[README.md](README.md) has the usage, [docs/design.md](docs/design.md) the design and the decisions with dates.

## Concept

### Two environments, one model bundle

```
User's trainer package                  |  package-openvino (this repository)
own code and dependencies               |  fixed code, users can't change it
                                        |
CV framework --export--> model bundle ==|==> model bundle --from_path()--> InferenceModel --> applications
                                        |         |                              ^
                                        |         +--> OpenVINO converter --> model bundle (OpenVINO IR)
```

- A user's trainer package trains with a CV framework (torchvision and RF-DETR today; Ultralytics, Darknet,
  D-FINE, mmdetection and custom models are not supported yet) and exports a **model bundle**.
- Only the bundle crosses the line: no Python code and no dependencies. **Loading a bundle never runs code
  from it.**
- This package is the same for every model. It converts a bundle to OpenVINO IR (FP32, FP16, INT8) and loads
  any bundle with `GraphInferenceModel.from_path()`. The applications are predict, visualize, benchmark latency
  and benchmark accuracy. Serving is not built yet.
- Today the export step lives here too, as `model_exporters`. It is the only package that depends on the
  frameworks, and in the concept it belongs on the trainer package side. A bundle with a torch network needs
  `model_exporters` to rebuild the network, so the bundles that cross the line are ONNX or OpenVINO bundles.

### Definition and implementation

A bundle is a zip (or directory) with `model_info.json` and the files of each component. `model_info.json`
describes the model as a graph of components that read and write named items of one data dictionary:

```
Definition (in the bundle):   load_image -> resize_image -> image_to_tensor -> onnx_network -> anchor_nms -> detections_to_bboxes
                                  :              :                :                 :              :                :
Implementation (torch):     TorchLoadImage  TorchResizeImage  TorchImageToTensor  OnnxRuntimeNetwork  TorchAnchorNMS  TorchDetectionsToBboxes
```

- **Definition** (`inference_model_core`): one pydantic model per component type. Its fields are the
  parameters, and `input_ports` / `output_ports` declare the ports. It holds no implementation.
- **Implementation** (`inference_model_torch` and the inference engine packages): `Component[<definition>]`
  with a `forward` whose parameters are the input ports. Registration fails when they don't match.
- **Graph engine**: torch is the only one. Components exchange `torch.Tensor`s, which stay on the device of
  the component that produced them. ONNX and OpenVINO networks convert to NumPy at their boundary.
- **Conversion** replaces the network component and keeps the rest: torch → ONNX → OpenVINO → INT8.

## Packages

Each top-level package under `src/` is a separate concept that can later move to its own repository.

| Package | Role |
|---|---|
| `inference_model_core` | `ModelBundle` (model info schema, read, write, convert), component definitions, `GraphInferenceModel`, `load_graph` |
| `inference_model_torch` | The torch graph engine: `TorchModelGraph`, implementations of the standard definitions, network base classes, torch → ONNX |
| `inference_model_onnxruntime` | `onnx_network` on ONNX Runtime |
| `inference_model_openvino` | `openvino_network` on OpenVINO, `onnx_network` as an alternative, converters to OpenVINO IR and INT8 |
| `model_exporters` | Model families: network definitions, torch implementations, and the functions that write bundles |
| `package_openvino` | Trainer glue: checkpoints, datasets, latency, metrics, class mappings |

## Rules

- The user interface is two classes: `ModelBundle` (`from_path`, `from_zip`, `write`, `write_zip`, `convert`)
  and `GraphInferenceModel.from_path`. Don't add module-level functions that do the same.
- `inference_model_core` imports no torch, no graph engine and no inference engine. A test checks this.
- The inference engine packages import `inference_model_core` and `inference_model_torch`, never each other.
- A component type has one definition and one default implementation. An alternative implementation is
  registered as `<type>/<inference engine>` and chosen with `inference_engine=` at load time.
- Implementations and converters outside the standard components are registered as entry points in
  `pyproject.toml` (`hafnia.inference_model_components`, `hafnia.inference_model_converters`). Run `uv sync`
  after changing them.
- Values in the data dictionary are per sample, without a batch axis. A `batched` component gets them stacked.
- The bundle format has a version (`MODEL_INFO_FORMAT_VERSION`). Bundles of another major version are
  rejected, so a change to the schema needs a new version.
- A new model family needs a parity test against the original model.
- Update `docs/design.md` when the design changes, and add the decision to its section 11.

## Commands

```bash
uv sync --all-extras                           # Install everything incl. the dev group
uv run pytest                                  # All tests, about 7 minutes
uv run pytest -m "not slow"                    # Skip the end-to-end export tests
uv run pytest tests/inference_model_torch      # One module (one tests folder per package under src/)
uv run ruff check src tests scripts examples   # Lint
uv run ruff format src tests scripts examples  # Format (line length 120)
uv run mypy src tests scripts examples         # Type check
```

Tests use the micro dataset in `tests/data` and run offline, apart from the first download of pretrained
weights. Test folders have no `__init__.py`, so test file names are unique across folders.
