# Design review: package-openvino

Date: 2026-11-18
Reviewed: [design.md](design.md) and `src/inference_model_core/` (steps 1–2), before step 3 starts.

> **Status:** points 1, 2, 3, 4.1 and the 4.5 `uint8` fix were applied on 2026-11-18, together with
> step 3 (see design.md section 11). 4.2 (FP16) was applied with step 4. The other points are open.

## Verdict

The overall shape is sound and I would keep it: a zip bundle with `model_info.json`, a base
class that extends Hafnia's `InferenceModel`, and a registry that picks the implementation from
model info.

Three things should change before step 3, because step 3 fixes the export contract and the
first bundles get written:

1. **What we export from torchvision.** The plan exports the complete torchvision model, with
   resize and NMS inside the graph. I measured that this works, but it is a poor fit for the
   goals of this project (quantization, latency numbers, non-CPU devices). Export the network
   only and do thresholding and NMS in the flavor.
2. **The engine is stored in the bundle.** That ties a model file to one engine, so the same
   ONNX file can't be run on both ONNX Runtime and OpenVINO.
3. **Preprocessing has two possible sources of truth.** Quantization calibration uses the
   generic helper, while each flavor is free to preprocess its own way. If they differ, the
   model is calibrated on the wrong data and nothing fails visibly.

The rest are schema fixes that are cheap now and breaking later, points to raise with Intel, and
small code issues.

---

## 1. Export the network, not the whole torchvision model

### What I measured

Setup: `ssdlite320_mobilenet_v3_large` (pretrained), torch 2.10.0, torchvision 0.25.0,
`torch.onnx.export(..., dynamo=False, opset_version=17)`, onnxruntime 1.26.0, OpenVINO 2026.2.1,
NNCF 3.2.0, CPU of this laptop. Latency is the median of 30 runs on a random-noise image.

| | Full model (resize + NMS in graph) | Network only (backbone + head) |
|---|---|---|
| Input | `[3, H, W]`, no batch axis | `[N, 3, 320, 320]`, dynamic batch |
| Output | `boxes [n, 4]`, `scores [n]`, `labels [n]`, `n` varies per image | `bbox_regression [N, 3234, 4]`, `cls_logits [N, 3234, 91]` |
| Matches torch | Yes, at three input sizes (score diff ≤ 1e-6) | Yes (diff ≤ 3e-5) |
| OpenVINO CPU latency, FP32 | **33 ms** with `score_thresh=0.001`, **8.6 ms** with `score_thresh=0.5` | **6.6 ms** |
| OpenVINO CPU latency, INT8 | not measured | 4.8–5.5 ms |
| NNCF quantization | Runs when loaded through `ov.convert_model`. Fails with an internal error when the `.onnx` is loaded through `core.read_model`. | Runs (113 quantizer nodes) |
| Batching | Not possible | Works |

What this shows:

- **The score threshold is baked into the graph at export time, and it dominates latency.**
  Accuracy benchmarking needs a very low threshold (mAP uses all detections), serving needs a
  normal one. With the full model, one bundle can't serve both, and the "model latency" we report
  is mostly NMS: 33 ms versus 6.6 ms for the network itself.
- **Quantization is fragile on the full model.** It worked through one loading path and crashed
  through another. NNCF also has to step around an `If` node and the NMS subgraph.
- **No batching and no static shapes** with the full model. The design doc already lists static
  shapes as a requirement for NPU and a preference for NNCF.

Not conclusive: OpenVINO's GPU plugin on this laptop picks up the NVIDIA card, which is not a
supported device. The full model failed to compile there (no kernel for `NonZero`), and the
network-only model compiled but failed at inference with an OpenCL error. This needs a rerun on
Intel hardware before anything is concluded about GPU.

The INT8 run was calibrated on random noise, so it says the quantization runs and how fast the
result is. It says nothing about accuracy.

### Recommendation

For SSD-style (single-stage) models, define the bundle contract as:

- **Input:** `[N, 3, H, W]` with a fixed `H × W` and a dynamic batch axis.
- **Preprocessing:** entirely outside the graph and entirely described in model info (stretch
  resize to 320 × 320, scale, mean/std). `in_graph` is then rarely needed.
- **Output:** raw predictions per anchor. Thresholding, NMS and top-k happen in the flavor.
  Box decoding can go either side; decide in step 3 with a parity test.

What this buys: batching works, quantization touches only the network, latency numbers are
comparable with OpenVINO's `benchmark_app`, the threshold becomes a runtime setting, and shapes
are static.

What it costs: the flavor has to implement anchor decoding and NMS in NumPy (roughly 100 lines),
and resizing outside the graph (OpenCV) differs slightly from torchvision's in-graph resize, so
parity with the original torch model is "within tolerance" unless the torch model is fed an image
already resized to 320 × 320.

**Faster R-CNN is different.** It is two-stage: NMS runs in the middle of the network (on region
proposals), and its resize keeps the aspect ratio (`min_size=320, max_size=640`), so there is no
single fixed input size. It can only be exported as a full model. It exports and runs on ONNX
Runtime and OpenVINO CPU, but expect no batching, fragile quantization and CPU only. I would keep
it as the stress test you chose it to be, with its own decoder (`torchvision_detection`, full
graph), and not let it define the default contract.

---

## 2. Don't bind a bundle to one engine

`runtime.inference_engine` is stored in `model_info.json` and is part of the dispatch key. The
design doc says format and engine are separate on purpose because one `.onnx` file can run on
ONNX Runtime, OpenVINO and TensorRT. But with the engine stored in the file, running the same
ONNX bundle on OpenVINO means rewriting the bundle.

Recommendation:

- Keep the field, but treat it as the **default** engine.
- `inference_model_from_path(path, engine=None, ...)` lets the caller override it.
- Each flavor declares which `model_format`s it can load.

This also gives a useful debugging tool: running the same ONNX file on both engines separates
"the conversion changed the model" from "the engines disagree".

---

## 3. One implementation of preprocessing

You chose independent flavor classes per engine (Q4). That stands, but one consequence needs a
guard. Today:

- The generic OpenVINO exporter calibrates INT8 quantization with `preprocess_image()` driven by
  `PreprocessingInfo`.
- Each flavor implements its own `preprocess()` and may ignore `PreprocessingInfo`.

If those two differ (interpolation, colour order, padding value), the model is calibrated on
inputs it never sees at inference. Accuracy drops and no test fails unless a parity test happens
to cover it.

Recommendation: `InferenceModelBase.preprocess()` gets a **default implementation** built from
`PreprocessingInfo`. Flavors override it only for a stated reason. Calibration and inference then
run the same code. A flavor is reduced to engine I/O plus postprocessing, which also removes most
of the duplication that Q4 accepted.

---

## 4. Schema fixes (cheap now, breaking later)

**4.1 Separate user-tunable settings from model facts.** `load_model(path, **runtime_options)` is
an untyped bag that will end up mixing engine settings (`device`, threads) with decoder settings
(score threshold, max detections). Model info should hold defaults for the decoder settings, and
loading should accept typed overrides. The RF-DETR trainer already has this split
(`InferenceConfig.threshold`).

**4.2 Record precision truthfully.** `ov.save_model` compresses weights to FP16 by default. I
measured it: the `.bin` is 7.1 MB by default and 13.7 MB with `compress_to_fp16=False`. The
existing RF-DETR exporter therefore produces FP16 models without saying so. Our exporter should
pass the flag explicitly and write the matching `precision`.

**4.3 Open up the engine and format fields.** `inference_engine` and `model_format` are closed
`Literal` types in the core module. The registry is designed so a separately installed package
can add an engine, but that package would still need a core change to get its name accepted. Use
plain strings with constants for the known values.

**4.4 Drop `framework` from the dispatch key.** The key is `(engine, framework, decoder)`. The
decoder already defines the input/output contract, and `framework` only says where the model came
from. Keeping it in the key means a generic decoder (for example classification with softmax)
needs a separate flavor for torchvision, timm and our own classification trainer. I would use
`(engine, decoder)` and keep `framework` as information. This revisits Q3.

**4.5 Gaps in `PreprocessingInfo`.**

- No centre crop. The standard ImageNet evaluation transform is resize to 256 then centre-crop
  to 224, so the first classification model will need it.
- `scale` defaults to 1/255, but `dtype="uint8"` forbids scaling, so `PreprocessingInfo(dtype="uint8")`
  fails validation unless `scale=None` is also passed.
- The field names are close to, but not the same as, the OpenVINO Model API metadata that the
  existing exporter writes into the IR (`mean_values`, `scale_values`, `reverse_input_channels`,
  `resize_type`). Aligning the vocabulary makes writing that metadata mechanical. See 5.1.

**4.6 Add file hashes.** A `sha256` per model file in model info gives integrity checking and
makes `provenance.parent` meaningful.

---

## 5. Points to raise with Intel

**5.1 Overlap with OpenVINO Model API.** Model API already stores preprocessing and labels as
metadata inside the IR and has per-model-type wrappers that return typed results. That is the
same idea as our model info plus flavors, limited to OpenVINO. Worth asking Intel directly:

- Should our OpenVINO bundles be loadable by Model API, DL Streamer and OVMS as they are? The
  RF-DETR exporter already writes this metadata for DL Streamer.
- Should the OpenVINO flavor reuse Model API wrappers where one exists?

I am working from the metadata keys used in the existing exporter. Check the current Model API
documentation before committing to a mapping.

**5.2 Embedding preprocessing in the IR.** OpenVINO can build resize, colour conversion and
normalization into the model, so it accepts raw `uint8` images. Intel usually recommends this and
it is often faster. If the OpenVINO exporter does it, the ONNX and OpenVINO bundles of the same
model have different preprocessing. That works with the current schema (`in_graph`) and is one
more reason for point 3.

**5.3 Latency method.** Agree up front what is being compared. `benchmark_app` measures the
network only, and our `predict()` also includes preprocessing and postprocessing. Report the
stages separately and state the score threshold. Numbers from this laptop's GPU are not
meaningful for Intel hardware.

---

## 6. Smaller findings in the code

| # | Finding | Where | Suggested fix |
|---|---|---|---|
| 6.1 | Flavors only register when their module is imported, and the entry points in `pyproject.toml` are commented out. `inference_model_from_path` in a script will find nothing. | `pyproject.toml`, `registry.py` | Enable the entry points for the in-repo flavor modules in step 3. |
| 6.2 | The benchmark is meant to time stages by calling `preprocess` / `predict_raw` / `postprocess` itself. It would then re-implement the batching logic and could measure a different path than `predict_batch` runs. | `inference_model.py` | Record per-stage timings inside the base class. |
| 6.3 | A flavor that forgets `inference_engine`, `framework` or `decoder` fails with an `AttributeError` at registration. | `inference_model.py` | Check in `__init_subclass__` and raise a clear error. |
| 6.4 | `supports_batching()` only recognises a rank-4 input with a dynamic batch axis. A graph with a fixed batch size above 1 is run one image at a time and will fail. | `inference_model.py` | Document it, or reject such bundles on load. |
| 6.5 | `load_image_rgb` assumes NumPy images are RGB. Hafnia's `Sample.read_image()` returns BGR for samples stored in video files, and the raw PIL mode for images (palette PNGs become index arrays). This affects every `InferenceModel`, not only ours. | `hafnia_dataset_types.py:953` | Fix in Hafnia: always return RGB. |
| 6.6 | No size limit when reading zips. Once user-supplied ONNX is allowed, ONNX external-data paths are a known path-traversal route. | `bundle.py` | Add limits and validate external data before enabling bring-your-own ONNX. |
| 6.7 | Default torch wheels on Linux include CUDA, which adds several GB to the Docker image. Export is CPU tracing. | `pyproject.toml` | Use the CPU-only torch index. |
| 6.8 | `design.md` section 5 shows the torchvision input as `[1, 3, null, null]`. The real full-model input is `[3, H, W]`. | `docs/design.md` | Update once point 1 is decided. |
| 6.9 | Parity tests by index are flaky: detections with near-equal scores swap order between engines. I saw this with FP16 and with random-init weights. | step 3 tests | Match detections by IoU and class, then compare scores. |

---

## 7. What I would keep as is

- Zip bundle with `model_info.json` at the root, also loadable from a directory.
- Identity fields at the top, everything an export rewrites under `runtime`.
- `ModelBundleInfo` as a subclass in this package first, upstreamed to Hafnia later.
- Extending Hafnia's `InferenceModel`, and `get_model_info()` returning `ModelInfo`.
- The registry with entry-point discovery.
- Output-index to class mapping in decoder parameters, leaving `TaskInfo` unchanged.
- Pre- and postprocessing per image, with only the engine call batched.
- Image-only input and own-exporter bundles only in v1.
