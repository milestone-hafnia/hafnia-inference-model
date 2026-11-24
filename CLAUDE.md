# hafnia-inference-model

The model bundle library: it writes bundles, converts them to OpenVINO and runs any bundle through one
`InferenceModel` interface that returns Hafnia primitives. It is a member of the package-openvino uv workspace
(`../`) until it moves to its own repository. [README.md](README.md) has the usage,
[docs/design.md](docs/design.md) the design and the decisions with dates.

## Concept

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
- **Graph engine**: torch runs every graph, and there is no choice of graph engine. Components exchange
  `torch.Tensor`s, which stay on the device of the component that produced them. ONNX and OpenVINO networks convert to NumPy at their boundary.
- **Conversion** replaces the network component and keeps the rest: torch → ONNX → OpenVINO → INT8.
- **Export**: `model_exporters` writes bundles of the supported frameworks (torchvision and RF-DETR today). It
  is the only package that depends on the frameworks, and in the concept it belongs on the side of a user's
  trainer package. A bundle with a torch network needs `model_exporters` to rebuild the network.

## Packages

Each top-level package under `src/` is a separate concept that can later move to its own repository.

| Package | Role |
|---|---|
| `inference_model_core` | `ModelBundle` (model info schema, read, write, convert), component definitions, `GraphInferenceModel` (incl. accuracy and latency benchmarks), `load_graph` |
| `inference_model_torch` | The torch graph engine: `TorchModelGraph`, implementations of the standard definitions, network base classes, torch → ONNX |
| `inference_model_onnxruntime` | `onnx_network` on ONNX Runtime |
| `inference_model_openvino` | `openvino_network` on OpenVINO, `onnx_network` as an alternative, converters to OpenVINO IR and INT8 |
| `model_exporters` | Model families: network definitions, torch implementations, and the functions that write bundles |

## Rules

- Nothing here imports `package_openvino` or knows about the trainer package's scripts.
- The user interface is two classes, `ModelBundle` and `GraphInferenceModel`. New functionality is a method
  on one of them, not a module-level function that does the same.
- Every function that writes a bundle takes one path and returns the written `ModelBundle`. A path ending with
  `.zip` is written as a zip, any other as a folder.
- The load settings (`device`, `params`, `inference_engine`, ...) are defined once, in `LoadSettings`, and
  passed on as `**load_settings`. Add a setting there, not as an argument of each function.
- `inference_model_core` imports no torch and no inference engine when it is imported. A test checks this.
  Only `load_graph` imports the torch graph engine, inside the function.
- The inference engine packages import `inference_model_core` and `inference_model_torch`, never each other.
- A component type has one definition and one default implementation. An alternative implementation is
  registered as `<type>/<inference engine>` and chosen with `inference_engine=` at load time.
- Implementations and converters outside the standard components are registered as entry points in
  this folder's `pyproject.toml` (`hafnia.inference_model_components`, `hafnia.inference_model_converters`),
  and only there: no `@register_component` or `@register_converter` on them. Run `uv sync` after changing them.
- A new torch network needs no converter entry: `*/onnx` converts every torch network to ONNX.
- A model family that reads trainer archives has a `create_trainer_archive_bundle` function in the module its
  pretrained entry points name. There is no table of trainer models.
- A pretrained model is one entry point per model name (`hafnia.inference_model_pretrained_models`), behind
  `ModelBundle.from_pretrained`. A model family is imported only when one of its models is asked for, so a
  new family never adds a dependency to the other models.
- Values in the data dictionary are per sample, without a batch axis. A `batched` component gets them stacked.
- The bundle format has a version (`MODEL_INFO_FORMAT_VERSION`). Bundles of another major version are
  rejected, so a change to the schema needs a new version.
- A new model family needs a parity test against the original model.
- Update `docs/design.md` when the design changes, and add the decision to its "Decisions" section.
- Each fact has one home. How to use it: the README usage block and the docstrings. How it works and why:
  `docs/design.md`. Don't copy signatures or file trees into the documents, as copies go stale.

## Commands

Run them in this folder. `uv sync` in the workspace root installs the library with all extras and the dev group.

```bash
uv run pytest                           # All tests, about 7 minutes
uv run pytest -m "not slow"             # Skip the end-to-end export tests
uv run pytest tests/inference_model_torch  # One module (one tests folder per package under src/)
uv run ruff check src tests examples    # Lint
uv run ruff format src tests examples   # Format (line length 120)
uv run mypy src tests examples          # Type check
```

Tests use the micro dataset in `tests/data` and run offline, apart from the first download of pretrained
weights. Test folders have no `__init__.py`, so test file names are unique across folders.
