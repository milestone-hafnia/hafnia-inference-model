"""Test helpers: a dummy flavor that runs without any inference engine installed, and detection matching.

The dummy "model" is a JSON file with fixed detections in model input pixel coordinates. 'predict_raw'
returns those detections for every image in the batch and records the batch sizes it was called with.
"""

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from pydantic import BaseModel, ConfigDict

from inference_model_core import (
    InferenceModelBase,
    InputInfo,
    ModelBundleInfo,
    PostprocessingInfo,
    PreprocessingInfo,
    PreprocessMeta,
    RawTensors,
    ResizeInfo,
    RuntimeInfo,
    TensorInfo,
    read_bundle_file,
    read_model_info,
    write_bundle,
)
from inference_model_core.primitives import bboxes_from_input_xyxy

PATH_MICRO_COCO = Path(__file__).parent / "data" / "micro-coco-2017"

DUMMY_CLASS_NAMES = ["car", "person"]
DUMMY_INPUT_SIZE = 32
DUMMY_DETECTIONS = {
    # Pixel coordinates in the (32 x 32) model input
    "boxes": [[0.0, 0.0, 16.0, 16.0], [8.0, 8.0, 32.0, 24.0]],
    "scores": [0.9, 0.4],
    "labels": [0, 1],
}


def make_model_info(batch_dim: Optional[int] = None, file_path: str = "model.json") -> ModelBundleInfo:
    return ModelBundleInfo(
        name="dummy-detector",
        framework="dummy",
        variant="tiny",
        tasks=[TaskInfo.from_class_names(primitive=Bbox, class_names=DUMMY_CLASS_NAMES, name="object_detection")],
        runtime=RuntimeInfo(
            model_format="onnx",
            inference_engine="onnxruntime",
            file_path=file_path,
            input=InputInfo(
                tensors=[TensorInfo(name="images", shape=[batch_dim, 3, DUMMY_INPUT_SIZE, DUMMY_INPUT_SIZE])]
            ),
            outputs=[
                TensorInfo(name="boxes", shape=[batch_dim, None, 4]),
                TensorInfo(name="scores", shape=[batch_dim, None]),
                TensorInfo(name="labels", shape=[batch_dim, None], dtype="int64"),
            ],
            preprocessing=PreprocessingInfo(
                resize=ResizeInfo(mode="letterbox"), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
            postprocessing=PostprocessingInfo(decoder="dummy_detection", params={"score_threshold": 0.0}),
        ),
    )


class DummyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_threshold: float = 0.0


class DummyDetector(InferenceModelBase):
    inference_engine = "onnxruntime"
    framework = "dummy"
    decoder = "dummy_detection"
    supported_formats: ClassVar[Tuple[str, ...]] = ("onnx",)
    params_model = DummyParams
    params: DummyParams

    def __init__(
        self, model_info: ModelBundleInfo, detections: dict, postprocessing_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(model_info, postprocessing_params=postprocessing_params)
        self.detections = detections
        self.raw_batch_sizes: List[int] = []

    @classmethod
    def load_model(cls, path_bundle, postprocessing_params=None, **engine_options: Any) -> "DummyDetector":
        model_info = read_model_info(path_bundle)
        detections = json.loads(read_bundle_file(path_bundle, model_info.runtime.file_path))
        return cls(model_info=model_info, detections=detections, postprocessing_params=postprocessing_params)

    def write_model(self, path_bundle, overwrite: bool = False) -> Path:
        files = {self.model_info.runtime.file_path: json.dumps(self.detections).encode()}
        return write_bundle(path_bundle, self.model_info, files, overwrite=overwrite)

    def predict_raw(self, inputs: RawTensors) -> RawTensors:
        batch_size = inputs["images"].shape[0]
        self.raw_batch_sizes.append(batch_size)
        return {
            "boxes": np.tile(np.asarray(self.detections["boxes"], dtype=np.float32), (batch_size, 1, 1)),
            "scores": np.tile(np.asarray(self.detections["scores"], dtype=np.float32), (batch_size, 1)),
            "labels": np.tile(np.asarray(self.detections["labels"], dtype=np.int64), (batch_size, 1)),
        }

    def postprocess(self, outputs: RawTensors, meta: PreprocessMeta) -> List[Primitive]:
        return bboxes_from_input_xyxy(  # type: ignore[return-value]
            boxes_xyxy=outputs["boxes"][0],
            scores=outputs["scores"][0],
            class_indices=outputs["labels"][0],
            task=self.model_info.get_task_by_name("object_detection"),
            meta=meta,
            score_threshold=self.params.score_threshold,
        )


def write_dummy_bundle(path_bundle: Path, batch_dim: Optional[int] = None) -> Path:
    model = DummyDetector(model_info=make_model_info(batch_dim=batch_dim), detections=DUMMY_DETECTIONS)
    return model.write_model(path_bundle)


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
