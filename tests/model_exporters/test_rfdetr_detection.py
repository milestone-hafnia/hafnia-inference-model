"""RF-DETR -> ONNX bundle, checked against RF-DETR's own 'predict' (pretrained and a trainer archive)."""

import json
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import assert_detections_match

from inference_model_core import inference_model_from_path, read_model_info

pytest.importorskip("rfdetr")
pytest.importorskip("onnxruntime")
from model_exporters.registry import export_trainer_archive
from model_exporters.rfdetr_detection import DEFAULT_SCORE_THRESHOLD

pytestmark = pytest.mark.slow

RESOLUTION = 384  # RF-DETR Nano


def rfdetr_detections(model, categories: List[str], image: np.ndarray) -> List[Bbox]:
    """RF-DETR's 'predict' as normalized 'Bbox' primitives, without placeholder classes."""
    detections = model.predict(image, threshold=DEFAULT_SCORE_THRESHOLD)
    height, width = image.shape[:2]
    return [
        Bbox(
            top_left_x=x0 / width,
            top_left_y=y0 / height,
            width=(x1 - x0) / width,
            height=(y1 - y0) / height,
            class_name=categories[class_id],
            confidence=float(score),
        )
        for (x0, y0, x1, y1), class_id, score in zip(
            detections.xyxy, detections.class_id, detections.confidence, strict=True
        )
        if not categories[class_id].startswith("NotDefined_")
    ]


def test_model_info(rfdetr, micro_coco):
    info = read_model_info(rfdetr[2])
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()

    assert (info.framework, info.runtime.postprocessing.decoder) == ("rfdetr", "rfdetr_detection")
    # One class per logit column: COCO ids, with placeholders for id 0 and the unused ids. Same as torchvision's.
    class_names = info.tasks[0].get_class_names()
    assert len(class_names) == 91 and class_names[0] == "NotDefined_000"
    assert [name for name in class_names if not name.startswith("NotDefined_")] == coco_class_names
    assert info.runtime.input.tensors[0].shape == [None, 3, RESOLUTION, RESOLUTION]
    assert info.runtime.quantization_hints.model_type == "transformer"


def test_matches_rfdetr_without_resize(rfdetr, images):
    """Images already at the model input size: resizing is an identity on both sides, so the only differences
    are numerical (ONNX Runtime vs torch, NumPy vs torch postprocessing). Checked in both directions.

    Scores match to 1e-5. A few low-score boxes move a little (IoU down to 0.97 below score 0.2 on the micro
    dataset), so boxes are compared from score 0.2.
    """
    torch_model, categories, path_bundle = rfdetr
    model = inference_model_from_path(path_bundle)
    settings = {"min_score": 0.2, "min_iou": 0.99, "max_score_diff": 1e-3}
    n_checked = 0
    for image in images:
        image_resized = cv2.resize(image, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        reference, predicted = rfdetr_detections(torch_model, categories, image_resized), model.predict(image_resized)
        n_checked += assert_detections_match(reference, predicted, **settings)
        assert_detections_match(predicted, reference, **settings)
    assert n_checked >= len(images)


def test_matches_rfdetr_on_original_images(rfdetr, images):
    """Our antialiased resize runs on uint8 pixels, RF-DETR's on floats: confident detections still match."""
    torch_model, categories, path_bundle = rfdetr
    model = inference_model_from_path(path_bundle)
    n_checked = sum(
        assert_detections_match(
            rfdetr_detections(torch_model, categories, image),
            model.predict(image),
            min_score=0.3,
            min_iou=0.95,
            max_score_diff=0.05,
        )
        for image in images
    )
    assert n_checked >= len(images)


def test_trainer_archive(rfdetr, tmp_path, images):
    """A trainer-object-detection archive with the pretrained weights gives the same detections as the
    pretrained export and the same task."""
    torch_model, categories, path_pretrained_bundle = rfdetr
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=categories, name="object_detection")
    path_weights = Path(torch_model.model_config.pretrain_weights)
    config = {"name": "RFDETRNano", "task": task.model_dump(mode="json"), "model_weight_path": path_weights.name}
    path_archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(path_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("model_config.json", json.dumps(config))
        archive.write(path_weights, arcname=path_weights.name)

    path_bundle = export_trainer_archive(path_archive, tmp_path / "bundle.zip")
    assert read_model_info(path_bundle).tasks == read_model_info(path_pretrained_bundle).tasks
    expected = inference_model_from_path(path_pretrained_bundle).predict(images[0])
    predicted = inference_model_from_path(path_bundle).predict(images[0])
    assert assert_detections_match(expected, predicted, min_score=0.05, min_iou=0.999, max_score_diff=1e-4) > 0
