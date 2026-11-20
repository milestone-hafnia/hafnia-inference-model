"""The standard NumPy components: thin wrappers with typed parameters around the core functions."""

import numpy as np
import pytest
from hafnia.dataset.primitives import Bbox, Classification
from helpers import make_model_info, random_image
from PIL import Image
from pydantic import ValidationError

from inference_model_core import ImageTransform, list_components
from inference_model_core.components import (
    AnchorNMS,
    ClassificationFromLogits,
    DetectionsToBboxes,
    ImageToTensor,
    LoadImage,
    QueryTopK,
    ResizeImage,
)
from inference_model_core.detection import select_anchor_detections, select_query_detections

MODEL_INFO = make_model_info()  # Task 'object_detection' with the classes 'car' and 'person'


def create(component_class, inputs, outputs, params=None, model_info=MODEL_INFO):
    info = component_class.create_info(
        "step", inputs={p: p for p in inputs}, outputs={p: p for p in outputs}, params=params
    )
    return component_class(info, model_info)


def test_standard_components_are_registered():
    types = {key.type for key in list_components() if key.runtime == "numpy"}
    assert types >= {
        "load_image",
        "resize_image",
        "image_to_tensor",
        "anchor_nms",
        "query_topk",
        "detections_to_bboxes",
        "classification_from_logits",
    }


def test_image_components(tmp_path):
    image = random_image(20, 40)
    path_image = tmp_path / "image.png"
    Image.fromarray(image).save(path_image)

    loaded = create(LoadImage, ["image"], ["image"]).run({"image": str(path_image)})["image"]
    np.testing.assert_array_equal(loaded, image)

    resize = create(ResizeImage, ["image"], ["image", "transform"], {"mode": "letterbox", "height": 32, "width": 32})
    resized = resize.run({"image": loaded})
    assert resized["image"].shape == (32, 32, 3)
    assert isinstance(resized["transform"], ImageTransform) and resized["transform"].pad_top == 8

    to_tensor = create(ImageToTensor, ["image"], ["tensor"], {"mean": [0.5] * 3, "std": [0.5] * 3})
    tensor = to_tensor.run({"image": resized["image"]})["tensor"]
    assert tensor.shape == (3, 32, 32) and tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[:, 0, 0], -1.0)  # Padding: (0 / 255 - 0.5) / 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mean": [0.5, 0.5, 0.5]}, "both be set"),
        ({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5]}, "same length"),
        ({"mean": [0.5], "std": [0.0]}, "zeros"),
        ({"dtype": "uint8", "scale": 1 / 255}, "can not be scaled"),
        ({"layout": "NCHW"}, "layout"),
    ],
)
def test_invalid_image_to_tensor_params(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        ImageToTensor.create_info("to_tensor", inputs={"image": "a"}, outputs={"tensor": "b"}, params=kwargs)


def test_uint8_tensor_has_no_default_scale():
    info = ImageToTensor.create_info(
        "to_tensor", inputs={"image": "a"}, outputs={"tensor": "b"}, params={"dtype": "uint8"}
    )
    assert info.params["scale"] is None
    assert info.params["layout"] == "CHW"  # Defaults are written out, so the model info is self-describing


def test_ports_are_checked():
    with pytest.raises(ValueError, match=r"Missing inputs: \['image'\], unknown inputs: \['picture'\]"):
        create(ImageToTensor, ["picture"], ["tensor"])
    with pytest.raises(ValueError, match=r"unknown outputs: \['array'\]"):
        create(ImageToTensor, ["image"], ["array"])
    info = LoadImage.create_info("load", inputs={"image": "a"}, outputs={"image": "b"})
    with pytest.raises(ValueError, match="implements type 'resize_image'"):
        ResizeImage(info, MODEL_INFO)


def test_anchor_nms():
    rng = np.random.default_rng(0)
    corners = rng.uniform(0, 200, size=(50, 2))
    boxes = np.concatenate([corners, corners + rng.uniform(10, 80, size=(50, 2))], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, size=(50, 3)).astype(np.float32)
    params = {
        "score_threshold": 0.3,
        "nms_iou_threshold": 0.5,
        "topk_candidates": 20,
        "max_detections": 10,
        "background_index": 0,
    }
    outputs = create(AnchorNMS, ["boxes", "scores"], ["boxes", "scores", "labels"], params).run(
        {"boxes": boxes, "scores": scores}
    )

    expected = select_anchor_detections(boxes, scores, **params)
    for name, expected_values in zip(("boxes", "scores", "labels"), expected, strict=True):
        np.testing.assert_array_equal(outputs[name], expected_values)
    assert 0 < len(outputs["scores"]) <= 10 and 0 not in outputs["labels"]


def test_query_topk():
    rng = np.random.default_rng(0)
    boxes, logits = rng.uniform(0.2, 0.6, size=(30, 4)).astype(np.float32), rng.normal(size=(30, 5)).astype(np.float32)
    transform = ImageTransform(original_height=100, original_width=200, input_height=64, input_width=96)
    component = create(QueryTopK, ["boxes", "logits", "transform"], ["boxes", "scores", "labels"], {"num_select": 7})
    outputs = component.run({"boxes": boxes, "logits": logits, "transform": transform})

    expected = select_query_detections(boxes, logits, num_select=7, input_hw=(64, 96))
    for name, expected_values in zip(("boxes", "scores", "labels"), expected, strict=True):
        np.testing.assert_array_equal(outputs[name], expected_values)
    assert outputs["boxes"].shape == (7, 4)


def test_detections_to_bboxes_with_transform_or_image():
    detections = {
        "boxes": np.array([[0.0, 8.0, 16.0, 24.0], [0.0, 8.0, 32.0, 24.0], [4.0, 8.0, 8.0, 16.0]]),
        "scores": np.array([0.9, 0.2, 0.8]),
        "labels": np.array([1, 0, 2]),  # Label 2 is not a class of the task and is dropped
    }
    ports, params = ["boxes", "scores", "labels"], {"task_name": "object_detection", "score_threshold": 0.5}

    # Boxes in pixels of the 32 x 32 network input of a letterboxed 20 x 40 image
    transform = ImageTransform(20, 40, 32, 32, scale_x=0.8, scale_y=0.8, pad_top=8)
    component = create(DetectionsToBboxes, [*ports, "transform"], ["predictions"], params)
    (bbox,) = component.run({**detections, "transform": transform})["predictions"]
    assert isinstance(bbox, Bbox) and (bbox.class_name, bbox.task_name, bbox.ground_truth) == (
        "person",
        "object_detection",
        False,
    )
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.0, 0.5, 1.0))

    # Boxes in pixels of the original image, e.g. from a network that resizes itself
    component = create(DetectionsToBboxes, [*ports, "image"], ["predictions"], params)
    (bbox,) = component.run({**detections, "image": random_image(32, 64)})["predictions"]
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.25, 0.25, 0.5))

    for inputs in (ports, [*ports, "transform", "image"]):
        with pytest.raises(ValueError, match="needs either the 'transform' or the 'image' input"):
            create(DetectionsToBboxes, inputs, ["predictions"], params)
    with pytest.raises(ValueError, match="nothing"):
        create(DetectionsToBboxes, [*ports, "image"], ["predictions"], {"task_name": "nothing"})


def test_classification_from_logits():
    component = create(ClassificationFromLogits, ["logits"], ["predictions"], {"task_name": "object_detection"})
    (prediction,) = component.run({"logits": np.array([0.0, np.log(3.0)])})["predictions"]
    assert isinstance(prediction, Classification)
    assert (prediction.class_name, prediction.class_idx) == ("person", 1)
    assert prediction.confidence == pytest.approx(0.75)

    params = {"task_name": "object_detection", "score_threshold": 0.9}
    component = create(ClassificationFromLogits, ["logits"], ["predictions"], params)
    assert component.run({"logits": np.array([0.0, np.log(3.0)])})["predictions"] == []
