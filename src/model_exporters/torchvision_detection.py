"""Export torchvision detection models to ONNX model bundles.

Two export contracts (see docs/design.md and docs/design_review.md):

- SSD models (e.g. 'ssdlite320_mobilenet_v3_large') are exported as the **network only**: input
  '[N, 3, H, W]' with a dynamic batch axis, output per-anchor decoded boxes and class scores. Resizing and
  normalization are done outside the graph, thresholding/top-k/NMS by the flavor (decoder
  'anchor_boxes_nms'). The score threshold is then a runtime setting and the graph batches and quantizes.
- Faster R-CNN models are two-stage with NMS inside the network and an aspect-ratio preserving resize, so
  they are exported as the **full model**: input '[3, H, W]', output final boxes/scores/labels (decoder
  'boxes_scores_labels'). The graph is exported with a very low score threshold so the runtime threshold
  can only be raised.

This is the only module importing torch/torchvision.
"""

import copy
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import onnx
import torch
import torch.nn.functional as F
import torchvision
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from torch import nn
from torchvision.models import WeightsEnum, detection
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD

from inference_model_core import (
    InputInfo,
    ModelBundleInfo,
    PostprocessingInfo,
    PreprocessingInfo,
    ResizeInfo,
    RuntimeInfo,
    TensorInfo,
    write_bundle,
)

FRAMEWORK = "torchvision"
DECODER_ANCHOR_BOXES_NMS = "anchor_boxes_nms"
DECODER_BOXES_SCORES_LABELS = "boxes_scores_labels"
MODEL_FILENAME = "model.onnx"
TASK_NAME = "object_detection"
DEFAULT_OPSET_VERSION = 17
# Score threshold baked into full-graph exports. The runtime 'score_threshold' filters on top of it.
FULL_GRAPH_MIN_SCORE = 0.001
# Placeholder categories in torchvision's COCO label space that are not real classes
IGNORED_CATEGORIES = ("__background__", "N/A")

PretrainedModel = Tuple[Callable[..., nn.Module], WeightsEnum]
PRETRAINED_MODELS: Dict[str, PretrainedModel] = {
    "ssdlite320_mobilenet_v3_large": (
        detection.ssdlite320_mobilenet_v3_large,
        detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "fasterrcnn_mobilenet_v3_large_320_fpn": (
        detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
    ),
}


def load_pretrained_model(model_name: str) -> Tuple[nn.Module, List[str]]:
    """Build a pretrained torchvision detection model. Returns the model in eval mode and its categories
    (model label -> name, including torchvision's '__background__' and 'N/A' placeholders)."""
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Options: {list(PRETRAINED_MODELS)}")
    builder, weights = PRETRAINED_MODELS[model_name]
    return builder(weights=weights).eval(), list(weights.meta["categories"])


def export_pretrained_torchvision_detection(
    model_name: str,
    path_bundle: Union[str, Path],
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
) -> Path:
    model, categories = load_pretrained_model(model_name)
    return export_torchvision_detection(
        model,
        categories=categories,
        path_bundle=path_bundle,
        name=model_name,
        variant=model_name,
        opset_version=opset_version,
        overwrite=overwrite,
        provenance={"weights": str(PRETRAINED_MODELS[model_name][1])},
    )


def export_torchvision_detection(
    model: nn.Module,
    categories: List[str],
    path_bundle: Union[str, Path],
    name: str,
    variant: Optional[str] = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
    overwrite: bool = False,
    provenance: Optional[dict] = None,
) -> Path:
    """Export a torchvision detection model (pretrained or fine-tuned) as an ONNX model bundle.

    Args:
        model: An 'SSD' or 'FasterRCNN' model. It is not modified.
        categories: Category name per model label. '__background__' and 'N/A' entries are not task classes.
        path_bundle: Output bundle, a '.zip' path or a directory.
        name: Model name stored in the model info.
        variant: Informational variant stored in the model info.
        opset_version: ONNX opset.
        overwrite: Replace an existing bundle.
        provenance: Extra provenance information stored in the model info.
    """
    task, label_to_class_idx = task_from_categories(categories)
    provenance = {
        "exporter": __name__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opset_version": opset_version,
        **(provenance or {}),
    }

    with tempfile.TemporaryDirectory(prefix="torchvision_export_") as tmp_dir:
        path_onnx = Path(tmp_dir) / MODEL_FILENAME
        if isinstance(model, SSD):
            runtime = _export_ssd(model, path_onnx, label_to_class_idx, opset_version)
        elif isinstance(model, FasterRCNN):
            runtime = _export_full_graph(model, path_onnx, label_to_class_idx, opset_version)
        else:
            raise ValueError(f"Unsupported model type '{type(model).__name__}'. Supported: SSD, FasterRCNN.")
        onnx.checker.check_model(path_onnx.as_posix())

        runtime.provenance = {**provenance, **(runtime.provenance or {})}
        model_info = ModelBundleInfo(name=name, framework=FRAMEWORK, variant=variant, tasks=[task], runtime=runtime)
        return write_bundle(path_bundle, model_info, files={MODEL_FILENAME: path_onnx}, overwrite=overwrite)


def task_from_categories(categories: List[str]) -> Tuple[TaskInfo, List[int]]:
    """Task with the real classes, and the mapping from model label to task class index (-1 = ignored)."""
    class_names = [category for category in categories if category not in IGNORED_CATEGORIES]
    label_to_class_idx = []
    for category in categories:
        label_to_class_idx.append(-1 if category in IGNORED_CATEGORIES else class_names.index(category))
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=class_names, name=TASK_NAME)
    return task, label_to_class_idx


class _SSDNetwork(nn.Module):
    """SSD backbone + head with box decoding and softmax, for a fixed input size.

    Anchors are constant for a fixed input size and are baked into the graph. Box decoding mirrors
    'torchvision.models.detection._utils.BoxCoder.decode_single' followed by clipping to the image.
    """

    anchors: torch.Tensor  # (A, 4) xyxy anchors in input pixels, registered as buffer

    def __init__(self, ssd: SSD, height: int, width: int):
        super().__init__()
        self.backbone = ssd.backbone
        self.head = ssd.head
        self.weights = ssd.box_coder.weights
        self.bbox_xform_clip = ssd.box_coder.bbox_xform_clip
        self.height, self.width = height, width
        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width)
            features = list(ssd.backbone(dummy).values())
            anchors = ssd.anchor_generator(ImageList(dummy, [(height, width)]), features)[0]
        self.register_buffer("anchors", anchors)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        head_outputs = self.head(list(self.backbone(images).values()))
        boxes = self._decode(head_outputs["bbox_regression"])
        scores = F.softmax(head_outputs["cls_logits"], dim=-1)
        return boxes, scores

    def _decode(self, rel_codes: torch.Tensor) -> torch.Tensor:
        anchors = self.anchors
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[..., 0] / wx
        dy = rel_codes[..., 1] / wy
        dw = torch.clamp(rel_codes[..., 2] / ww, max=self.bbox_xform_clip)
        dh = torch.clamp(rel_codes[..., 3] / wh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x0 = (pred_ctr_x - 0.5 * pred_w).clamp(0, self.width)
        y0 = (pred_ctr_y - 0.5 * pred_h).clamp(0, self.height)
        x1 = (pred_ctr_x + 0.5 * pred_w).clamp(0, self.width)
        y1 = (pred_ctr_y + 0.5 * pred_h).clamp(0, self.height)
        return torch.stack([x0, y0, x1, y1], dim=-1)


def _export_ssd(ssd: SSD, path_onnx: Path, label_to_class_idx: List[int], opset_version: int) -> RuntimeInfo:
    transform = ssd.transform
    if transform.fixed_size is None:
        raise ValueError("Only SSD models with a fixed input size ('transform.fixed_size') are supported.")
    width, height = transform.fixed_size  # torchvision stores (width, height)
    network = _SSDNetwork(copy.deepcopy(ssd).eval(), height=height, width=width).eval()
    n_anchors, n_labels = network.anchors.shape[0], len(label_to_class_idx)

    dummy = torch.zeros(2, 3, height, width)
    _onnx_export(
        network,
        (dummy,),
        path_onnx,
        input_names=["images"],
        output_names=["boxes", "scores"],
        dynamic_axes={"images": {0: "batch"}, "boxes": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=opset_version,
    )

    return RuntimeInfo(
        model_format="onnx",
        inference_engine="onnxruntime",
        file_path=MODEL_FILENAME,
        precision="fp32",
        input=InputInfo(tensors=[TensorInfo(name="images", shape=[None, 3, height, width])]),
        outputs=[
            TensorInfo(name="boxes", shape=[None, n_anchors, 4]),
            TensorInfo(name="scores", shape=[None, n_anchors, n_labels]),
        ],
        # Same as torchvision's GeneralizedRCNNTransform: bilinear resize (no antialiasing) to the fixed
        # size, then normalization (the order doesn't matter as both are linear)
        preprocessing=PreprocessingInfo(
            color_order="RGB",
            resize=ResizeInfo(mode="stretch", height=height, width=width, interpolation="linear"),
            scale=1 / 255,
            mean=list(transform.image_mean),
            std=list(transform.image_std),
            layout="NCHW",
            dtype="float32",
        ),
        postprocessing=PostprocessingInfo(
            decoder=DECODER_ANCHOR_BOXES_NMS,
            params={
                "score_threshold": ssd.score_thresh,
                "nms_iou_threshold": ssd.nms_thresh,
                "topk_candidates": ssd.topk_candidates,
                "max_detections": ssd.detections_per_img,
                "background_index": 0,
                "label_to_class_idx": label_to_class_idx,
            },
        ),
    )


class _FullGraphDetector(nn.Module):
    """Single image in, final detections out. Returns tensors in a fixed order instead of a dict."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detections = self.model([image])[0]
        return detections["boxes"], detections["scores"], detections["labels"]


def _export_full_graph(
    model: FasterRCNN, path_onnx: Path, label_to_class_idx: List[int], opset_version: int
) -> RuntimeInfo:
    runtime_score_threshold = model.roi_heads.score_thresh
    export_model = copy.deepcopy(model).eval()
    export_model.roi_heads.score_thresh = min(FULL_GRAPH_MIN_SCORE, runtime_score_threshold)
    # Tracing bakes in the Python branch "proposals > 0" of torchvision's box decoding: an image without
    # proposals (e.g. a blank frame, rejected by the RPN score threshold) then crashes the graph. With an RPN
    # score threshold of 0 there are always proposals. All detections of the original model are kept; a few
    # extra, mostly low-score detections can appear from proposals the original model would have dropped.
    export_model.rpn.score_thresh = 0.0

    dummy = torch.rand(3, 480, 640)
    _onnx_export(
        _FullGraphDetector(export_model),
        (dummy,),
        path_onnx,
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "image": {1: "height", 2: "width"},
            "boxes": {0: "detections"},
            "scores": {0: "detections"},
            "labels": {0: "detections"},
        },
        opset_version=opset_version,
    )

    transform = model.transform
    return RuntimeInfo(
        model_format="onnx",
        inference_engine="onnxruntime",
        file_path=MODEL_FILENAME,
        precision="fp32",
        input=InputInfo(tensors=[TensorInfo(name="image", shape=[3, None, None])]),
        outputs=[
            TensorInfo(name="boxes", shape=[None, 4]),
            TensorInfo(name="scores", shape=[None]),
            TensorInfo(name="labels", shape=[None], dtype="int64"),
        ],
        # Resize (aspect ratio preserving) and normalization are part of the graph. Boxes come out in pixel
        # coordinates of the input image.
        preprocessing=PreprocessingInfo(
            color_order="RGB",
            scale=1 / 255,
            mean=list(transform.image_mean),
            std=list(transform.image_std),
            layout="NCHW",
            dtype="float32",
            in_graph=["resize", "normalize"],
        ),
        postprocessing=PostprocessingInfo(
            decoder=DECODER_BOXES_SCORES_LABELS,
            params={"score_threshold": runtime_score_threshold, "label_to_class_idx": label_to_class_idx},
        ),
        provenance={
            "graph_min_score": export_model.roi_heads.score_thresh,
            "graph_max_detections": export_model.roi_heads.detections_per_img,
            "rpn_score_threshold": export_model.rpn.score_thresh,
            "original_rpn_score_threshold": model.rpn.score_thresh,
        },
    )


def _onnx_export(module: nn.Module, args: tuple, path_onnx: Path, **kwargs) -> None:
    # The TorchScript-based exporter handles torchvision detection models (data dependent control flow
    # in the transform and NMS). Tracer and user warnings about these constructs are expected; the exported
    # graphs are verified against torch in tests/test_torchvision_onnxruntime.py.
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        torch.onnx.export(module.eval(), args, path_onnx.as_posix(), dynamo=False, do_constant_folding=True, **kwargs)
