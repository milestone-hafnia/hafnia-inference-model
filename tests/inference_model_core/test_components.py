"""Components are pydantic models: fields are the configuration, 'forward' the implementation, and its signature
names the ports. The standard NumPy components are thin wrappers around the core functions."""

from typing import NamedTuple, Optional

import numpy as np
import pytest
from hafnia.dataset.primitives import Bbox, Classification
from helpers import make_model_info, random_image
from PIL import Image
from pydantic import ValidationError

from inference_model_core import Component, ImageTransform, NumpyComponent, list_components
from inference_model_core.component import Ports, call_component, signature_ports
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


def load(component: Component, inputs, outputs=("predictions",)) -> Component:
    """The component as it is in a loaded graph, connected to data keys named as its ports."""
    info = component.connect("step", inputs={port: port for port in inputs}, outputs={port: port for port in outputs})
    return type(component).load(info, MODEL_INFO)


class Scaled(NamedTuple):
    scaled: np.ndarray
    factor: float


class Scale(NumpyComponent):
    type = "scale"
    factor: float = 2.0

    def forward(self, values: np.ndarray, offset: Optional[np.ndarray] = None) -> Scaled:
        return Scaled(values * self.factor + (0 if offset is None else offset), self.factor)


def test_ports_come_from_the_signature_of_forward():
    assert Scale().ports() == Ports(inputs=("values",), optional_inputs=("offset",), outputs=("scaled", "factor"))
    assert ResizeImage(mode="stretch", height=8, width=8).ports().outputs == ("image", "transform")
    assert DetectionsToBboxes(task_name="a").ports() == Ports(
        inputs=("boxes", "scores", "labels"), optional_inputs=("transform", "image"), outputs=("predictions",)
    )


def test_components_run_without_a_graph():
    outputs = Scale(factor=3).forward(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(outputs.scaled, [3.0, 6.0])
    assert call_component(Scale(), {"values": np.array([1.0]), "offset": np.array([1.0])}) == {
        "scaled": np.array([3.0]),
        "factor": 2.0,
    }
    with pytest.raises(RuntimeError, match="not part of a loaded graph"):
        Scale().info  # noqa: B018


def test_configuration_is_typed_fields():
    assert Scale().model_dump() == {"factor": 2.0}
    with pytest.raises(ValidationError, match="factor"):
        Scale(factor="fast")
    with pytest.raises(ValidationError, match="speed"):
        Scale(speed=1)
    with pytest.raises(ValidationError, match="frozen"):
        Scale().factor = 3.0


def test_connect_gives_the_component_info():
    info = Scale(factor=4).connect("scale", inputs={"values": "tensor"}, outputs={"scaled": "tensor_scaled"})
    assert (info.type, info.runtime, info.params) == ("scale", "numpy", {"factor": 4.0})
    assert Scale.load(info, MODEL_INFO, params={"factor": 5}).factor == 5.0  # With an override

    with pytest.raises(ValueError, match=r"Missing inputs: \['values'\], unknown inputs: \['numbers'\]"):
        Scale().connect("scale", inputs={"numbers": "a"}, outputs={"scaled": "b"})
    with pytest.raises(ValueError, match=r"unknown outputs: \['result'\]"):
        Scale().connect("scale", inputs={"values": "a"}, outputs={"result": "b"})
    with pytest.raises(ValueError, match="implements type 'resize_image'"):
        ResizeImage.load(info, MODEL_INFO)


def test_forward_signatures_that_can_not_name_ports():
    class NoNames(NumpyComponent):
        type = "no_names"

        def forward(self, *values: np.ndarray) -> Scaled:
            return Scaled(values[0], 1.0)

    class NoOutputNames(NumpyComponent):
        type = "no_output_names"

        def forward(self, values: np.ndarray) -> np.ndarray:
            return values

    class DefaultValue(NumpyComponent):
        type = "default_value"

        def forward(self, values: np.ndarray, factor: float = 2.0) -> Scaled:
            return Scaled(values, factor)

    with pytest.raises(TypeError, match="Input ports are named parameters"):
        signature_ports(NoNames)
    with pytest.raises(TypeError, match="must be annotated to return a 'NamedTuple'"):
        signature_ports(NoOutputNames)
    with pytest.raises(TypeError, match="default of the optional input 'factor' must be 'None'"):
        signature_ports(DefaultValue)
    with pytest.raises(TypeError, match="returned a 'ndarray'"):
        call_component(NoOutputNames(), {"values": np.zeros(1)})


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

    loaded = LoadImage().forward(str(path_image)).image
    np.testing.assert_array_equal(loaded, image)

    resized = ResizeImage(mode="letterbox", height=32, width=32).forward(loaded)
    assert resized.image.shape == (32, 32, 3)
    assert isinstance(resized.transform, ImageTransform) and resized.transform.pad_top == 8

    tensor = ImageToTensor(mean=[0.5] * 3, std=[0.5] * 3).forward(resized.image).tensor
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
def test_invalid_image_to_tensor_configuration(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        ImageToTensor(**kwargs)


def test_uint8_tensor_has_no_default_scale():
    info = ImageToTensor(dtype="uint8").connect("to_tensor", inputs={"image": "a"}, outputs={"tensor": "b"})
    assert info.params["scale"] is None
    assert info.params["layout"] == "CHW"  # Defaults are written out, so the model info is self-describing


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
    detections = AnchorNMS(**params).forward(boxes, scores)

    expected = select_anchor_detections(boxes, scores, **params)
    for values, expected_values in zip(detections, expected, strict=True):
        np.testing.assert_array_equal(values, expected_values)
    assert 0 < len(detections.scores) <= 10 and 0 not in detections.labels


def test_query_topk():
    rng = np.random.default_rng(0)
    boxes, logits = rng.uniform(0.2, 0.6, size=(30, 4)).astype(np.float32), rng.normal(size=(30, 5)).astype(np.float32)
    transform = ImageTransform(original_height=100, original_width=200, input_height=64, input_width=96)
    detections = QueryTopK(num_select=7).forward(boxes, logits, transform)

    expected = select_query_detections(boxes, logits, num_select=7, input_hw=(64, 96))
    for values, expected_values in zip(detections, expected, strict=True):
        np.testing.assert_array_equal(values, expected_values)
    assert detections.boxes.shape == (7, 4)


def test_detections_to_bboxes_with_transform_or_image():
    detections = {
        "boxes": np.array([[0.0, 8.0, 16.0, 24.0], [0.0, 8.0, 32.0, 24.0], [4.0, 8.0, 8.0, 16.0]]),
        "scores": np.array([0.9, 0.2, 0.8]),
        "labels": np.array([1, 0, 2]),  # Label 2 is not a class of the task and is dropped
    }
    ports = ["boxes", "scores", "labels"]
    to_bboxes = DetectionsToBboxes(task_name="object_detection", score_threshold=0.5)

    # Boxes in pixels of the 32 x 32 network input of a letterboxed 20 x 40 image
    transform = ImageTransform(20, 40, 32, 32, scale_x=0.8, scale_y=0.8, pad_top=8)
    (bbox,) = load(to_bboxes, [*ports, "transform"]).forward(**detections, transform=transform).predictions
    assert isinstance(bbox, Bbox)
    assert (bbox.class_name, bbox.task_name, bbox.ground_truth) == ("person", "object_detection", False)
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.0, 0.5, 1.0))

    # Boxes in pixels of the original image, e.g. from a network that resizes itself
    (bbox,) = load(to_bboxes, [*ports, "image"]).forward(**detections, image=random_image(32, 64)).predictions
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.25, 0.25, 0.5))

    for inputs in (ports, [*ports, "transform", "image"]):
        with pytest.raises(ValueError, match="needs either the 'transform' or the 'image' input"):
            load(to_bboxes, inputs)
    with pytest.raises(ValueError, match="nothing"):
        load(DetectionsToBboxes(task_name="nothing"), [*ports, "image"])


def test_classification_from_logits():
    logits = np.array([0.0, np.log(3.0)])
    component = load(ClassificationFromLogits(task_name="object_detection"), ["logits"])
    (prediction,) = component.forward(logits).predictions
    assert isinstance(prediction, Classification)
    assert (prediction.class_name, prediction.class_idx) == ("person", 1)
    assert prediction.confidence == pytest.approx(0.75)

    component = load(ClassificationFromLogits(task_name="object_detection", score_threshold=0.9), ["logits"])
    assert component.forward(logits).predictions == []
