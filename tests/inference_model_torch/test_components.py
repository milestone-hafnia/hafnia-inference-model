"""Implementations compute a component definition on the torch graph engine: 'forward' takes the input ports of
the definition and returns the outputs by port name."""

from importlib.metadata import EntryPoint
from typing import Any, Dict, Optional

import pytest
import torch
from hafnia.dataset.primitives import Bbox, Classification
from helpers import make_bundle, random_image
from PIL import Image

from inference_model_core import ComponentDefinition, ImageTransform, Port
from inference_model_core.definitions import (
    AnchorNMS,
    ClassificationFromLogits,
    DetectionsToBboxes,
    ImageToTensor,
    LoadImage,
    QueryTopK,
    ResizeImage,
)
from inference_model_torch import Component, get_component_class, list_components, register_component
from inference_model_torch.component import ENTRY_POINT_GROUP, find_alternative_class, unregister_component
from inference_model_torch.components import TorchResizeImage
from inference_model_torch.components.detection import select_anchor_detections, select_query_detections

MODEL_INFO = make_bundle()  # Task 'object_detection' with the classes 'car' and 'person'


def load(definition: ComponentDefinition, inputs=None, outputs=None) -> Component:
    """The implementation of a definition as it is in a loaded graph, connected to data keys named as its ports."""
    ports = definition.ports()
    info = definition.connect(
        "step",
        inputs={port: port for port in (inputs if inputs is not None else ports.inputs)},
        outputs={port: port for port in (outputs if outputs is not None else ports.outputs)},
    )
    return get_component_class(info.type).load(info, MODEL_INFO)


class Scale(ComponentDefinition):
    type = "scale"
    input_ports = (Port("values"), Port("offset", optional=True))
    output_ports = (Port("scaled"),)

    factor: float = 2.0


def test_an_implementation_names_its_definition():
    class TorchScale(Component[Scale]):
        def forward(self, values: torch.Tensor, offset: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            return {"scaled": values * self.params.factor + (0 if offset is None else offset)}

    assert TorchScale.definition is Scale
    assert (TorchScale.engine, TorchScale.key()) == ("torch", "scale/torch")
    assert (TorchResizeImage.definition, TorchResizeImage.key()) == (ResizeImage, "resize_image/torch")

    register_component(TorchScale)
    try:
        assert "scale" in list_components()
        component = load(Scale(factor=3), inputs=["values"])
        assert isinstance(component, TorchScale) and component.params == Scale(factor=3)
        assert component.forward(torch.tensor([1.0, 2.0]))["scaled"].tolist() == [3.0, 6.0]
        assert TorchScale.load(component.info, MODEL_INFO, params={"factor": 5}).params.factor == 5.0
    finally:
        unregister_component(TorchScale)


def test_forward_must_take_the_ports_of_the_definition():
    class WrongInput(Component[Scale]):
        def forward(self, numbers: torch.Tensor, offset: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            return {"scaled": numbers}

    class OptionalNotOptional(Component[Scale]):
        def forward(self, values: torch.Tensor, offset: torch.Tensor) -> Dict[str, Any]:
            return {"scaled": values}

    class NoDefinition(Component):
        def forward(self, values: torch.Tensor) -> Dict[str, Any]:
            return {"scaled": values}

    for component_class in (WrongInput, OptionalNotOptional):
        with pytest.raises(TypeError, match=r"'Scale' declares the inputs \['values'\] and the optional inputs"):
            register_component(component_class)
    with pytest.raises(TypeError, match="names no definition"):
        register_component(NoDefinition)


class PackagedScale(Component[Scale]):
    """As a component of an installed package: no decorator, it is listed as an entry point."""

    def forward(self, values: torch.Tensor, offset: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        return {"scaled": values * self.params.factor}


class WrongPackagedScale(Component[Scale]):
    engine = "other"

    def forward(self, numbers: torch.Tensor) -> Dict[str, Any]:
        return {"scaled": numbers}


def test_components_of_packages_are_registered_by_their_entry_point(monkeypatch):
    entries = {
        "scale": EntryPoint("scale", f"{__name__}:PackagedScale", ENTRY_POINT_GROUP),
        "scale/other": EntryPoint("scale/other", f"{__name__}:WrongPackagedScale", ENTRY_POINT_GROUP),
    }
    monkeypatch.setattr("inference_model_torch.component._entry_point_components", lambda: entries)
    try:
        assert "scale" in list_components()  # Listed without importing it
        assert get_component_class("scale") is PackagedScale
        assert get_component_class("scale") is PackagedScale  # Registered by the first lookup
        # The class is checked like a decorated one, when it is first needed
        with pytest.raises(TypeError, match=r"'Scale' declares the inputs \['values'\]"):
            find_alternative_class("scale", "other")
    finally:
        unregister_component(PackagedScale)


def test_standard_components_are_implemented():
    assert set(list_components()) >= {
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

    loaded = load(LoadImage()).forward(str(path_image))["image"]
    assert torch.equal(loaded, torch.from_numpy(image))

    resized = load(ResizeImage(mode="letterbox", height=32, width=32)).forward(loaded)
    assert resized["image"].shape == (32, 32, 3) and resized["image"].dtype == torch.uint8
    assert isinstance(resized["transform"], ImageTransform) and resized["transform"].pad_top == 8

    tensor = load(ImageToTensor(mean=[0.5] * 3, std=[0.5] * 3)).forward(resized["image"])["tensor"]
    assert tensor.shape == (3, 32, 32) and tensor.dtype == torch.float32
    assert torch.allclose(tensor[:, 0, 0], torch.tensor(-1.0))  # Padding: (0 / 255 - 0.5) / 0.5


def test_resize_checks_what_torch_supports():
    with pytest.raises(ValueError, match="torch only supports 'antialias' for 'linear' and 'cubic'"):
        load(ResizeImage(mode="stretch", height=8, width=8, interpolation="area", antialias=True))


def test_anchor_nms():
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(50, 2, generator=generator) * 200
    boxes = torch.cat([corners, corners + 10 + torch.rand(50, 2, generator=generator) * 70], dim=1)
    scores = torch.rand(50, 3, generator=generator)
    params = {
        "score_threshold": 0.3,
        "nms_iou_threshold": 0.5,
        "topk_candidates": 20,
        "max_detections": 10,
        "background_index": 0,
    }
    detections = load(AnchorNMS(**params)).forward(boxes, scores)

    expected = select_anchor_detections(boxes, scores, **params)
    for port, expected_values in zip(("boxes", "scores", "labels"), expected, strict=True):
        assert torch.equal(detections[port], expected_values)
    assert 0 < len(detections["scores"]) <= 10 and 0 not in detections["labels"]


def test_query_topk():
    generator = torch.Generator().manual_seed(0)
    boxes, logits = torch.rand(30, 4, generator=generator) * 0.4 + 0.2, torch.randn(30, 5, generator=generator)
    transform = ImageTransform(original_height=100, original_width=200, input_height=64, input_width=96)
    detections = load(QueryTopK(num_select=7)).forward(boxes, logits, transform)

    expected = select_query_detections(boxes, logits, num_select=7, input_hw=(64, 96))
    for port, expected_values in zip(("boxes", "scores", "labels"), expected, strict=True):
        assert torch.equal(detections[port], expected_values)
    assert detections["boxes"].shape == (7, 4)


def test_detections_to_bboxes_with_transform_or_image():
    detections = {
        "boxes": torch.tensor([[0.0, 8.0, 16.0, 24.0], [0.0, 8.0, 32.0, 24.0], [4.0, 8.0, 8.0, 16.0]]),
        "scores": torch.tensor([0.9, 0.2, 0.8]),
        "labels": torch.tensor([1, 0, 2]),  # Label 2 is not a class of the task and is dropped
    }
    ports = ["boxes", "scores", "labels"]
    to_bboxes = DetectionsToBboxes(task_name="object_detection", score_threshold=0.5)

    # Boxes in pixels of the 32 x 32 network input of a letterboxed 20 x 40 image
    transform = ImageTransform(20, 40, 32, 32, scale_x=0.8, scale_y=0.8, pad_top=8)
    (bbox,) = load(to_bboxes, [*ports, "transform"]).forward(**detections, transform=transform)["predictions"]
    assert isinstance(bbox, Bbox)
    assert (bbox.class_name, bbox.task_name, bbox.ground_truth) == ("person", "object_detection", False)
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.0, 0.5, 1.0))

    # Boxes in pixels of the original image, e.g. from a network that resizes itself
    image = torch.from_numpy(random_image(32, 64))
    (bbox,) = load(to_bboxes, [*ports, "image"]).forward(**detections, image=image)["predictions"]
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.25, 0.25, 0.5))

    with pytest.raises(ValueError, match="nothing"):  # The task is looked up when the graph is loaded
        load(DetectionsToBboxes(task_name="nothing"), [*ports, "image"])


def test_classification_from_logits():
    logits = torch.tensor([1.0, 3.0]).log()
    component = load(ClassificationFromLogits(task_name="object_detection"))
    (prediction,) = component.forward(logits)["predictions"]
    assert isinstance(prediction, Classification)
    assert (prediction.class_name, prediction.class_idx) == ("person", 1)
    assert prediction.confidence == pytest.approx(0.75)

    component = load(ClassificationFromLogits(task_name="object_detection", score_threshold=0.9))
    assert component.forward(logits)["predictions"] == []
