"""Model bundles of torchvision detection models.

The bundle is a graph of components with a torch network (defined in 'model_exporters.definitions'), which
converts to ONNX and from there to OpenVINO (see 'inference_model_core.convert'). Two kinds of networks (see
docs/design.md):

- SSD models (e.g. 'ssdlite320_mobilenet_v3_large') are the **network only** ('torchvision_ssd_network'): input
  '[N, 3, H, W]', output per-anchor decoded boxes and class scores. Resizing and normalization run before the
  network, thresholding/top-k/NMS after it ('anchor_nms'). The score threshold is then a runtime setting and the
  network batches and quantizes:

      load_image -> resize -> to_tensor -> network -> nms -> to_bboxes

- Faster R-CNN models are two-stage with NMS inside the network and an aspect-ratio preserving resize, so they
  are the **full model** ('torchvision_faster_rcnn'): input '[3, H, W]', output final boxes/scores/labels in
  pixels of the image. The network runs with a very low score threshold so the runtime threshold can only be
  raised:

      load_image -> to_tensor -> network -> to_bboxes
"""

import contextlib
import copy
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from torch import nn
from torchvision.models import WeightsEnum, detection
from torchvision.models.detection import transform as detection_transform
from torchvision.models.detection._utils import BoxCoder
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
from torchvision.ops.misc import FrozenBatchNorm2d

from inference_model_core import (
    ComponentInfo,
    ConversionInfo,
    GraphInfo,
    ModelBundleInfo,
    NetworkParams,
    TensorInfo,
    write_bundle,
)
from inference_model_core.convert import TARGET_ONNX, TARGET_OPENVINO, TARGET_OPENVINO_INT8
from inference_model_core.definitions import AnchorNMS, DetectionsToBboxes, ImageToTensor, LoadImage, ResizeImage
from inference_model_torch import register_component
from inference_model_torch.network import OnnxExport, TorchNetwork, load_state_dict_file
from inference_model_torch.onnx_export import DEFAULT_OPSET_VERSION, batch_axes, onnx_export, onnx_options
from model_exporters.class_names import detection_task, placeholder_class_name
from model_exporters.definitions import TorchvisionFasterRCNN, TorchvisionSSDNetwork

FRAMEWORK = "torchvision"
WEIGHTS_FILE = "network/weights.pth"
TASK_NAME = "object_detection"
# Score threshold of full-model networks. The runtime 'score_threshold' filters on top of it.
FULL_MODEL_MIN_SCORE = 0.001
# Placeholder categories in torchvision's COCO label space, renamed to unique placeholder class names
PLACEHOLDER_CATEGORIES = ("__background__", "N/A")

# Graph parts to keep in floating point when quantizing, as ONNX node name patterns (option
# 'ignored_node_patterns' of the 'openvino_int8' conversion). Only the network (convolutions and linear layers)
# is quantized. Box decoding, softmax, resizing and NMS stay in float: 8 bit pixel coordinates and scores cost
# accuracy, and NNCF quantizing the SSD decoding also made the scores so noisy that NMS got 30x slower.
SSD_INT8_OPTIONS = {"ignored_node_patterns": [r"/(?!backbone/|head/).*"]}
FASTER_RCNN_INT8_OPTIONS = {
    "ignored_node_patterns": [
        r"/model/transform/.*",
        r"/model/rpn/(?!head/).*",
        r"/model/roi_heads/(?!box_head/|box_predictor/).*",
    ]
}

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


def create_pretrained_torchvision_detection_bundle(
    model_name: str, path_bundle: Union[str, Path], overwrite: bool = False
) -> Path:
    model, categories = load_pretrained_model(model_name)
    return create_torchvision_detection_bundle(
        model,
        architecture=model_name,
        categories=categories,
        path_bundle=path_bundle,
        name=model_name,
        variant=model_name,
        overwrite=overwrite,
        provenance={"weights": str(PRETRAINED_MODELS[model_name][1])},
    )


def create_torchvision_detection_bundle(
    model: nn.Module,
    architecture: str,
    categories: List[str],
    path_bundle: Union[str, Path],
    name: str,
    variant: Optional[str] = None,
    overwrite: bool = False,
    provenance: Optional[dict] = None,
) -> Path:
    """Write a torchvision detection model (pretrained or fine-tuned) as a model bundle with a torch network.

    Args:
        model: An 'SSD' or 'FasterRCNN' model. It is not modified.
        architecture: torchvision architecture of 'model' (see 'PRETRAINED_MODELS'). The bundle stores the
            weights and rebuilds the module from this name.
        categories: Category name per model label. '__background__' and 'N/A' entries get placeholder names.
        path_bundle: Output bundle, a '.zip' path or a directory.
        name: Model name stored in the model info.
        variant: Informational variant stored in the model info.
        overwrite: Replace an existing bundle.
        provenance: Extra provenance information stored with the network component.
    """
    if architecture not in PRETRAINED_MODELS:
        raise ValueError(f"Unsupported architecture '{architecture}'. Options: {list(PRETRAINED_MODELS)}")
    task = task_from_categories(categories)
    provenance = {
        "created_by": __name__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        **(provenance or {}),
    }
    if isinstance(model, SSD):
        components = _ssd_components(model, architecture, n_labels=len(categories), provenance=provenance)
    elif isinstance(model, FasterRCNN):
        components = _faster_rcnn_components(model, architecture, n_labels=len(categories), provenance=provenance)
    else:
        raise ValueError(f"Unsupported model type '{type(model).__name__}'. Supported: SSD, FasterRCNN.")

    model_info = ModelBundleInfo(
        name=name,
        framework=FRAMEWORK,
        variant=variant,
        tasks=[task],
        graph=GraphInfo(inputs=["image"], outputs=["predictions"], components=components),
    )
    with tempfile.TemporaryDirectory(prefix="torchvision_bundle_") as tmp_dir:
        path_weights = Path(tmp_dir) / "weights.pth"
        torch.save(model.state_dict(), path_weights)
        return write_bundle(path_bundle, model_info, files={WEIGHTS_FILE: path_weights}, overwrite=overwrite)


def _network_conversions(onnx_options: Dict[str, Any], int8_options: Dict[str, Any]) -> Dict[str, ConversionInfo]:
    return {
        TARGET_ONNX: ConversionInfo(options=onnx_options),
        TARGET_OPENVINO: ConversionInfo(),
        TARGET_OPENVINO_INT8: ConversionInfo(options=int8_options),
    }


def _ssd_components(ssd: SSD, architecture: str, n_labels: int, provenance: dict) -> List[ComponentInfo]:
    transform = ssd.transform
    if transform.fixed_size is None:
        raise ValueError("Only SSD models with a fixed input size ('transform.fixed_size') are supported.")
    width, height = transform.fixed_size  # torchvision stores (width, height)
    nms = AnchorNMS(
        score_threshold=ssd.score_thresh,
        nms_iou_threshold=ssd.nms_thresh,
        topk_candidates=ssd.topk_candidates,
        max_detections=ssd.detections_per_img,
        background_index=0,
    )
    return [
        LoadImage().connect("load_image", inputs={"image": "image"}, outputs={"image": "image_rgb"}),
        # Same as torchvision's GeneralizedRCNNTransform: bilinear resize (no antialiasing) to the fixed size,
        # then normalization (the order doesn't matter as both are linear)
        ResizeImage(mode="stretch", height=height, width=width, interpolation="linear").connect(
            "resize", inputs={"image": "image_rgb"}, outputs={"image": "image_resized", "transform": "image_transform"}
        ),
        ImageToTensor(scale=1 / 255, mean=list(transform.image_mean), std=list(transform.image_std)).connect(
            "to_tensor", inputs={"image": "image_resized"}, outputs={"tensor": "input_tensor"}
        ),
        TorchvisionSSDNetwork(architecture=architecture, num_classes=n_labels).connect(
            "network",
            inputs={"images": "input_tensor"},
            outputs={"boxes": "anchor_boxes", "scores": "anchor_scores"},
            files=[WEIGHTS_FILE],
            batched=True,
            conversions=_network_conversions(
                {"opset_version": DEFAULT_OPSET_VERSION, "dynamic_batch": True}, SSD_INT8_OPTIONS
            ),
            provenance=provenance,
        ),
        nms.connect(
            "nms",
            inputs={"boxes": "anchor_boxes", "scores": "anchor_scores"},
            outputs={"boxes": "boxes", "scores": "scores", "labels": "labels"},
        ),
        DetectionsToBboxes(task_name=TASK_NAME).connect(
            "to_bboxes",
            inputs={"boxes": "boxes", "scores": "scores", "labels": "labels", "transform": "image_transform"},
            outputs={"predictions": "predictions"},
        ),
    ]


def _faster_rcnn_components(
    model: FasterRCNN, architecture: str, n_labels: int, provenance: dict
) -> List[ComponentInfo]:
    roi_heads = model.roi_heads
    network = TorchvisionFasterRCNN(
        architecture=architecture,
        num_classes=n_labels,
        min_score=min(FULL_MODEL_MIN_SCORE, roi_heads.score_thresh),
        nms_iou_threshold=roi_heads.nms_thresh,
        max_detections=roi_heads.detections_per_img,
    )
    return [
        LoadImage().connect("load_image", inputs={"image": "image"}, outputs={"image": "image_rgb"}),
        # Resize (aspect ratio preserving) and normalization are part of the network
        ImageToTensor(scale=1 / 255).connect(
            "to_tensor", inputs={"image": "image_rgb"}, outputs={"tensor": "input_tensor"}
        ),
        network.connect(
            "network",
            inputs={"image": "input_tensor"},
            outputs={"boxes": "boxes", "scores": "scores", "labels": "labels"},
            files=[WEIGHTS_FILE],
            batched=False,
            conversions=_network_conversions({"opset_version": DEFAULT_OPSET_VERSION}, FASTER_RCNN_INT8_OPTIONS),
            provenance=provenance,
        ),
        # Boxes come out in pixels of the image, so they are mapped with the image size
        DetectionsToBboxes(task_name=TASK_NAME, score_threshold=roi_heads.score_thresh).connect(
            "to_bboxes",
            inputs={"boxes": "boxes", "scores": "scores", "labels": "labels", "image": "image_rgb"},
            outputs={"predictions": "predictions"},
        ),
    ]


def task_from_categories(categories: List[str]) -> TaskInfo:
    """Task with one class per model label: '__background__' and the 'N/A' entries of torchvision's COCO
    categories get unique placeholder names ('NotDefined_000', 'NotDefined_012', ...)."""
    class_names = [
        placeholder_class_name(label) if category in PLACEHOLDER_CATEGORIES else category
        for label, category in enumerate(categories)
    ]
    return detection_task(class_names, TASK_NAME)


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


def _build_detector(architecture: str, num_classes: int, path_weights: Path) -> nn.Module:
    """Rebuild a torchvision detector and load its weights. No pretrained weights are downloaded."""
    if architecture not in PRETRAINED_MODELS:
        raise ValueError(f"Unsupported architecture '{architecture}'. Options: {list(PRETRAINED_MODELS)}")
    builder, _ = PRETRAINED_MODELS[architecture]
    model = builder(weights=None, weights_backbone=None, num_classes=num_classes)
    state_dict = load_state_dict_file(path_weights)
    # torchvision builds detectors with frozen batch norm when they start from pretrained weights, and with
    # regular batch norm otherwise. Frozen batch norm has no 'num_batches_tracked': rebuild what was saved.
    if not any(key.endswith("num_batches_tracked") for key in state_dict):
        _freeze_batch_norm(model)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _freeze_batch_norm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=child.eps))
        else:
            _freeze_batch_norm(child)


@register_component
class TorchvisionSSDModule(TorchNetwork[TorchvisionSSDNetwork]):
    def build_model(self, path_weights: Path, device: torch.device) -> nn.Module:
        ssd = _build_detector(self.params.architecture, self.params.num_classes, path_weights)
        if not isinstance(ssd, SSD) or ssd.transform.fixed_size is None:
            raise ValueError(f"'{self.params.architecture}' is not an SSD model with a fixed input size.")
        width, height = ssd.transform.fixed_size  # torchvision stores (width, height)
        return _SSDNetwork(ssd, height=height, width=width).to(device).eval()

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        boxes, scores = self.model(images)
        return {"boxes": boxes, "scores": scores}

    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        options = onnx_options(options, defaults={"opset_version": DEFAULT_OPSET_VERSION, "dynamic_batch": True})
        network = copy.deepcopy(self.model).cpu()
        height, width, dynamic_batch = network.height, network.width, options["dynamic_batch"]
        onnx_export(
            network,
            (torch.zeros(2 if dynamic_batch else 1, 3, height, width),),
            path_onnx,
            input_names=["images"],
            output_names=["boxes", "scores"],
            dynamic_axes=batch_axes(["images", "boxes", "scores"], dynamic_batch),
            opset_version=options["opset_version"],
        )
        batch, n_anchors = None if dynamic_batch else 1, network.anchors.shape[0]
        params = NetworkParams(
            input_tensors=[TensorInfo(name="images", shape=[batch, 3, height, width])],
            output_tensors=[
                TensorInfo(name="boxes", shape=[batch, n_anchors, 4]),
                TensorInfo(name="scores", shape=[batch, n_anchors, self.params.num_classes]),
            ],
        )
        return OnnxExport(params=params, batched=True)


class _FullGraphDetector(nn.Module):
    """Single image in, final detections out. Returns tensors in a fixed order instead of a dict."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detections = self.model([image])[0]
        return detections["boxes"], detections["scores"], detections["labels"]


class _ExportBoxCoder(BoxCoder):
    """'BoxCoder' whose 'decode' also exports correctly for zero boxes.

    torchvision's 'decode' reshapes with '-1' behind a Python 'if box_sum > 0'. Tracing records the branch,
    and in ONNX a reshape of zero boxes to '(0, -1, 4)' is ambiguous, so an image without proposals (e.g. a
    blank frame) crashed the exported graph. This 'decode' has no reshape: it is 'decode_single' with the
    coordinates stacked to '(boxes, classes, 4)' directly. Same arithmetic as torchvision.
    """

    def decode(self, rel_codes: torch.Tensor, boxes: List[torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        reference = torch.cat(boxes, dim=0).to(rel_codes.dtype)
        widths = reference[:, 2] - reference[:, 0]
        heights = reference[:, 3] - reference[:, 1]
        ctr_x = reference[:, 0] + 0.5 * widths
        ctr_y = reference[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx  # (boxes, classes)
        dy = rel_codes[:, 1::4] / wy
        dw = torch.clamp(rel_codes[:, 2::4] / ww, max=self.bbox_xform_clip)
        dh = torch.clamp(rel_codes[:, 3::4] / wh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        half_w = 0.5 * torch.exp(dw) * widths[:, None]
        half_h = 0.5 * torch.exp(dh) * heights[:, None]
        corners = (pred_ctr_x - half_w, pred_ctr_y - half_h, pred_ctr_x + half_w, pred_ctr_y + half_h)
        return torch.stack(corners, dim=2)

    @classmethod
    def from_box_coder(cls, box_coder: BoxCoder) -> "_ExportBoxCoder":
        return cls(box_coder.weights, bbox_xform_clip=box_coder.bbox_xform_clip)


def _batched_nms_export(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float):
    """torchvision's 'batched_nms' (coordinate trick, as used when tracing) without its zero-boxes branch.

    That branch returns 'torch.empty((0,))', which becomes an empty constant in an ONNX 'If'. OpenVINO reads
    empty constants as scalars, so for an image without boxes it returned one element of uninitialized
    memory. Here the zero-boxes case takes the normal path: adding 0 to the maximum is a no-op for boxes
    clipped to the image (coordinates >= 0), so the result is unchanged otherwise.
    """
    max_coordinate = torch.cat([boxes.reshape(-1), boxes.new_zeros(1)]).max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    return nms(boxes + offsets[:, None], scores, iou_threshold)


def _clip_boxes_to_image_export(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """torchvision's 'clip_boxes_to_image' (tracing path) without the final 'reshape(boxes.shape)'.

    The exporter turns that reshape into '(boxes, classes, -1)', which is ambiguous for zero boxes. Stacking
    the four clipped coordinates gives the same result without a reshape.
    """
    height, width = size
    zero = torch.tensor(0, dtype=boxes.dtype, device=boxes.device)
    max_x = torch.tensor(width, dtype=boxes.dtype, device=boxes.device)
    max_y = torch.tensor(height, dtype=boxes.dtype, device=boxes.device)
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    clipped = [torch.min(torch.max(x, zero), max_x) for x in (x0, x1)]
    clipped += [torch.min(torch.max(y, zero), max_y) for y in (y0, y1)]
    return torch.stack([clipped[0], clipped[2], clipped[1], clipped[3]], dim=-1)


_original_resize_image_and_masks = detection_transform._resize_image_and_masks


def _resize_image_export(
    image: torch.Tensor,
    self_min_size: int,
    self_max_size: int,
    target: Optional[Dict[str, torch.Tensor]] = None,
    fixed_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """torchvision's '_resize_image_and_masks' (inference) with the output size computed in integers.

    torchvision scales by 'min(min_size / shortest side, max_size / longest side)' and floors 'side * scale'.
    Eager torch does that in double precision, the traced graph in float32, where an exact result such as
    400 * 0.8 = 320 can come out as 319.99998 and floor to 319. That shifted the detections of images whose
    resized size is an exact integer (e.g. 400 x 640). Here the size is 'side * numerator // denominator' in
    int64, which is exact.
    """
    if target is not None or fixed_size is not None:
        return _original_resize_image_and_masks(image, self_min_size, self_max_size, target, fixed_size)
    shape = torch._shape_as_tensor(image)[-2:]
    shortest, longest = shape.min(), shape.max()
    # The constraint that gives the smaller scale, compared without division
    use_min_size = self_min_size * longest <= self_max_size * shortest
    numerator = torch.where(use_min_size, torch.tensor(self_min_size), torch.tensor(self_max_size))
    denominator = torch.where(use_min_size, shortest, longest)
    size = torch.div(shape * numerator, denominator, rounding_mode="floor")
    resized = F.interpolate(image[None], size=[size[0], size[1]], mode="bilinear", align_corners=False)  # type: ignore[list-item]
    return resized[0], target


@contextlib.contextmanager
def _export_safe_ops() -> Iterator[None]:
    """Use export-safe versions of torchvision functions while exporting a two-stage detector.

    The detection modules call these through their modules ('torchvision.ops.boxes' and
    'torchvision.models.detection.transform'), so patching the module attributes is enough. Only active
    during export.
    """
    patches = [
        (box_ops, "batched_nms", _batched_nms_export),
        (box_ops, "clip_boxes_to_image", _clip_boxes_to_image_export),
        (detection_transform, "_resize_image_and_masks", _resize_image_export),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    for module, name, replacement in patches:
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


@register_component
class TorchvisionFasterRCNNModule(TorchNetwork[TorchvisionFasterRCNN]):
    def build_model(self, path_weights: Path, device: torch.device) -> nn.Module:
        params = self.params
        model = _build_detector(params.architecture, params.num_classes, path_weights)
        if not isinstance(model, FasterRCNN):
            raise ValueError(f"'{params.architecture}' is not a Faster R-CNN model.")
        model.roi_heads.score_thresh = params.min_score
        model.roi_heads.nms_thresh = params.nms_iou_threshold
        model.roi_heads.detections_per_img = params.max_detections
        return _FullGraphDetector(model).to(device).eval()

    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        boxes, scores, labels = self.model(image)
        return {"boxes": boxes, "scores": scores, "labels": labels}

    def export_onnx(self, path_onnx: Path, options: Dict[str, Any]) -> OnnxExport:
        options = onnx_options(options, defaults={"opset_version": DEFAULT_OPSET_VERSION})
        export_model = copy.deepcopy(self.model).cpu().eval()
        # torchvision handles images without proposals (e.g. blank frames, common in video) with Python branches
        # and reshapes that don't survive tracing, and computes the resize in float32 when traced. Replace those
        # parts for the export (see the helpers above).
        roi_heads = export_model.model.roi_heads
        roi_heads.box_coder = _ExportBoxCoder.from_box_coder(roi_heads.box_coder)

        with _export_safe_ops():
            onnx_export(
                export_model,
                (torch.rand(3, 480, 640),),
                path_onnx,
                input_names=["image"],
                output_names=["boxes", "scores", "labels"],
                dynamic_axes={
                    "image": {1: "height", 2: "width"},
                    "boxes": {0: "detections"},
                    "scores": {0: "detections"},
                    "labels": {0: "detections"},
                },
                opset_version=options["opset_version"],
            )
        params = NetworkParams(
            input_tensors=[TensorInfo(name="image", shape=[3, None, None])],
            output_tensors=[
                TensorInfo(name="boxes", shape=[None, 4]),
                TensorInfo(name="scores", shape=[None]),
                TensorInfo(name="labels", shape=[None], dtype="int64"),
            ],
        )
        return OnnxExport(params=params, batched=False)
