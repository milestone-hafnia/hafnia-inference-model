"""Test helpers: a dummy network component that runs without any inference engine installed, and detection
matching.

The dummy "network" is a JSON file with fixed detections in network input pixel coordinates. It returns those
detections for every sample and records the shapes of the inputs it was called with.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from pydantic import BaseModel, ConfigDict

from inference_model_core import (
    Component,
    ComponentInfo,
    ConversionInfo,
    GraphInfo,
    LoadOptions,
    ModelBundleInfo,
    write_bundle,
)
from inference_model_core.components import DetectionsToBboxes, ImageToTensor, LoadImage, ResizeImage

DUMMY_RUNTIME = "dummy"
DUMMY_CLASS_NAMES = ["car", "person"]
DUMMY_INPUT_SIZE = 32
DUMMY_NETWORK_FILE = "network/model.json"
DUMMY_DETECTIONS = {
    # Pixel coordinates in the (32 x 32) network input
    "boxes": [[0.0, 0.0, 16.0, 16.0], [8.0, 8.0, 32.0, 24.0]],
    "scores": [0.9, 0.4],
    "labels": [0, 1],
}


class DummyNetworkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_batch_size: Optional[int] = None


class DummyNetwork(Component):
    """A 'network' on the made-up runtime 'dummy'. Works batched and per sample."""

    type = "network"
    runtime = DUMMY_RUNTIME
    engine = DUMMY_RUNTIME
    params_model = DummyNetworkParams
    params: DummyNetworkParams
    input_ports = ("images",)
    output_ports = ("boxes", "scores", "labels")

    def __init__(self, info: ComponentInfo, model_info: ModelBundleInfo, detections: dict, options: LoadOptions):
        super().__init__(info, model_info)
        self.detections = detections
        self.options = options
        self.input_shapes: List[Tuple[int, ...]] = []

    @classmethod
    def load(cls, info, model_info, path_dir, options) -> "DummyNetwork":
        return cls(info, model_info, json.loads((path_dir / info.files[0]).read_text()), options)

    def files(self) -> Dict[str, Any]:
        return {self.info.files[0]: json.dumps(self.detections).encode()}

    @property
    def max_batch_size(self) -> Optional[int]:
        return self.params.max_batch_size

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.input_shapes.append(inputs["images"].shape)
        outputs: Dict[str, np.ndarray] = {
            "boxes": np.asarray(self.detections["boxes"], dtype=np.float32),
            "scores": np.asarray(self.detections["scores"], dtype=np.float32),
            "labels": np.asarray(self.detections["labels"], dtype=np.int64),
        }
        if not self.info.batched:
            return outputs
        batch_size = inputs["images"].shape[0]
        return {name: np.stack([value] * batch_size) for name, value in outputs.items()}


class DummyNetworkOnOtherEngine(DummyNetwork):
    """The same component type and runtime on another engine."""

    engine = "other"


def make_model_info(
    batched: bool = True,
    max_batch_size: Optional[int] = None,
    conversions: Optional[Dict[str, ConversionInfo]] = None,
    class_names: Optional[List[str]] = None,
) -> ModelBundleInfo:
    """image -> letterbox to 32 x 32 -> tensor -> dummy network -> bounding boxes.

    The network detects label 0 and label 1 on every image: by default the classes 'car' and 'person'."""
    size = DUMMY_INPUT_SIZE
    components = [
        LoadImage.create_info("load_image", inputs={"image": "image"}, outputs={"image": "image_rgb"}),
        ResizeImage.create_info(
            "resize",
            inputs={"image": "image_rgb"},
            outputs={"image": "image_resized", "transform": "image_transform"},
            params={"mode": "letterbox", "height": size, "width": size},
        ),
        ImageToTensor.create_info(
            "to_tensor",
            inputs={"image": "image_resized"},
            outputs={"tensor": "input_tensor"},
            params={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        ),
        DummyNetwork.create_info(
            "network",
            inputs={"images": "input_tensor"},
            outputs={"boxes": "boxes", "scores": "scores", "labels": "labels"},
            params={"max_batch_size": max_batch_size},
            files=[DUMMY_NETWORK_FILE],
            batched=batched,
            conversions=conversions,
        ),
        DetectionsToBboxes.create_info(
            "to_bboxes",
            inputs={"boxes": "boxes", "scores": "scores", "labels": "labels", "transform": "image_transform"},
            outputs={"predictions": "predictions"},
            params={"task_name": "object_detection"},
        ),
    ]
    return ModelBundleInfo(
        name="dummy-detector",
        framework="dummy",
        variant="tiny",
        tasks=[
            TaskInfo.from_class_names(
                primitive=Bbox, class_names=class_names or DUMMY_CLASS_NAMES, name="object_detection"
            )
        ],
        graph=GraphInfo(inputs=["image"], outputs=["predictions"], components=components),
    )


def network_of(model: Any) -> Component:
    """The 'network' component of a loaded inference model or model graph."""
    graph = getattr(model, "graph", model)
    return next(component for component in graph.components if component.info.name == "network")


def write_dummy_bundle(path_bundle: Path, **model_info_kwargs: Any) -> Path:
    files = {DUMMY_NETWORK_FILE: json.dumps(DUMMY_DETECTIONS).encode()}
    return write_bundle(path_bundle, make_model_info(**model_info_kwargs), files)


def random_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def bbox_iou(a: Bbox, b: Bbox) -> float:
    ax0, ay0, ax1, ay1 = a.top_left_x, a.top_left_y, a.top_left_x + a.width, a.top_left_y + a.height
    bx0, by0, bx1, by1 = b.top_left_x, b.top_left_y, b.top_left_x + b.width, b.top_left_y + b.height
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def assert_detections_match(
    reference: List[Bbox],
    predicted: List[Bbox],
    min_score: float,
    min_iou: float,
    max_score_diff: float,
) -> int:
    """Every reference detection with score >= 'min_score' must have a predicted detection of the same class
    with IoU >= 'min_iou' and a score within 'max_score_diff'. Matching by IoU instead of by index keeps the
    check stable when detections with near-equal scores swap order between engines.

    Returns the number of reference detections that were checked.
    """
    checked = [ref for ref in reference if ref.confidence >= min_score]
    for ref in checked:
        candidates = [p for p in predicted if p.class_name == ref.class_name]
        best = max(candidates, key=lambda p: bbox_iou(ref, p), default=None)
        assert best is not None, f"No prediction of class '{ref.class_name}' for reference {ref}"
        iou = bbox_iou(ref, best)
        assert iou >= min_iou, f"Best IoU {iou:.4f} < {min_iou} for reference {ref.class_name} ({ref.confidence:.3f})"
        score_diff = abs(ref.confidence - best.confidence)
        assert score_diff <= max_score_diff, f"Score differs by {score_diff:.4f} for {ref.class_name} (IoU {iou:.3f})"
    return len(checked)
