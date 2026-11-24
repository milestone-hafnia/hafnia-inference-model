"""Bundles of RF-DETR, checked against RF-DETR's own 'predict' (pretrained and a trainer archive): the bundle with
the torch network and its conversion to ONNX."""

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

from inference_model_core import GraphInferenceModel, ModelBundle, NetworkParams

pytest.importorskip("rfdetr")
pytest.importorskip("onnxruntime")
import torch

from model_exporters.registry import create_bundle_from_trainer_archive
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


def test_model_info(rfdetr_torch, rfdetr, micro_coco):
    torch_info, info = ModelBundle.from_path(rfdetr_torch[2]), ModelBundle.from_path(rfdetr[2])
    torch_network, network = torch_info.graph.get_component("network"), info.graph.get_component("network")
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()

    assert [c.type for c in torch_info.graph.components] == [
        "load_image",
        "resize_image",
        "image_to_tensor",
        "rfdetr_network",
        "query_topk",
        "detections_to_bboxes",
    ]
    assert (torch_network.type, torch_network.params) == ("rfdetr_network", {"model_name": "RFDETRNano"})
    assert info.framework == "rfdetr"
    # One class per logit column: COCO ids, with placeholders for id 0 and the unused ids. Same as torchvision's.
    class_names = info.tasks[0].get_class_names()
    assert len(class_names) == 91 and class_names[0] == "NotDefined_000"
    assert [name for name in class_names if not name.startswith("NotDefined_")] == coco_class_names

    assert (network.type, network.batched) == ("onnx_network", True)
    params = NetworkParams.model_validate(network.params)
    assert params.input_tensors[0].shape == [None, 3, RESOLUTION, RESOLUTION]
    assert [tensor.shape for tensor in params.output_tensors] == [[None, 300, 4], [None, 300, 91]]
    assert network.conversions["openvino_int8"].options == {"model_type": "transformer"}


def test_bundle_checkpoint_holds_only_weights_and_plain_settings(rfdetr_torch, tmp_path):
    """RF-DETR's checkpoint has the training state (three times the weights) and pickled objects. The bundle's
    copy has neither, so it loads without unpickling objects."""
    torch_model, _, path_bundle = rfdetr_torch
    path_weights = tmp_path / "weights.pth"
    path_weights.write_bytes(ModelBundle.from_path(path_bundle).read_file("network/weights.pth"))

    checkpoint = torch.load(path_weights, map_location="cpu", weights_only=True)
    assert sorted(checkpoint) == ["args", "model"]
    assert checkpoint["args"]["num_queries"] == 300
    assert path_weights.stat().st_size < 0.5 * Path(torch_model.model_config.pretrain_weights).stat().st_size


def test_torch_bundle_matches_rfdetr(rfdetr_torch, images):
    """The torch network is rebuilt from the model name and the bundle's checkpoint. Run on the device RF-DETR
    itself uses, the scores are the same to 1e-5 (CPU against CUDA differs by 1e-3)."""
    torch_model, categories, path_bundle = rfdetr_torch
    model = GraphInferenceModel.from_path(path_bundle, device=torch_model.model.device.type.upper())
    assert {type(component).engine for component in model.graph.components} == {"torch"}
    settings = {"min_score": 0.2, "min_iou": 0.99, "max_score_diff": 1e-4}
    n_checked = 0
    for image in images:
        image_resized = cv2.resize(image, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        reference, predicted = rfdetr_detections(torch_model, categories, image_resized), model.predict(image_resized)
        n_checked += assert_detections_match(reference, predicted, **settings)
        assert_detections_match(predicted, reference, **settings)
    assert n_checked >= len(images)


def test_matches_rfdetr_without_resize(rfdetr, images):
    """Images already at the model input size: resizing is an identity on both sides, so the only differences
    are numerical (ONNX Runtime vs torch, the bundle's vs the model's postprocessing). Checked in both directions.

    Scores match to 1e-5. A few low-score boxes move a little (IoU down to 0.97 below score 0.2 on the micro
    dataset), so boxes are compared from score 0.2.
    """
    torch_model, categories, path_bundle = rfdetr
    model = GraphInferenceModel.from_path(path_bundle)
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
    model = GraphInferenceModel.from_path(path_bundle)
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


def test_trainer_archive(rfdetr_torch, tmp_path, images):
    """A trainer-object-detection archive with the pretrained weights gives the same bundle as the pretrained
    model: the same task and the same detections."""
    torch_model, categories, path_pretrained_bundle = rfdetr_torch
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=categories, name="object_detection")
    path_weights = Path(torch_model.model_config.pretrain_weights)
    config = {"name": "RFDETRNano", "task": task.model_dump(mode="json"), "model_weight_path": path_weights.name}
    path_archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(path_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("model_config.json", json.dumps(config))
        archive.write(path_weights, arcname=path_weights.name)

    path_bundle = create_bundle_from_trainer_archive(path_archive, tmp_path / "bundle.zip")
    info, pretrained_info = ModelBundle.from_path(path_bundle), ModelBundle.from_path(path_pretrained_bundle)
    assert info.tasks == pretrained_info.tasks
    assert info.graph.get_component("network").provenance["trainer_archive"] == "checkpoint.zip"
    expected = GraphInferenceModel.from_path(path_pretrained_bundle).predict(images[0])
    predicted = GraphInferenceModel.from_path(path_bundle).predict(images[0])
    assert assert_detections_match(expected, predicted, min_score=0.05, min_iou=0.999, max_score_diff=1e-4) > 0
