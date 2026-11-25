"""Bundles of torchvision detection models, checked against the original torch models: the bundle with the torch
network as the bundle helper writes it, and its conversion to ONNX run on ONNX Runtime, the reference engine.

Parity is checked by matching detections on IoU and class (not by index), on the micro COCO images.
"""

import re
from typing import List

import cv2
import numpy as np
import pytest
from hafnia.dataset.primitives import Bbox
from helpers import assert_detections_match, network_of

from hafnia_inference_model.core import GraphInferenceModel, ModelBundle, NetworkParams

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
import onnx  # noqa: E402

from hafnia_inference_model.exporters.torchvision_detection import task_from_categories  # noqa: E402

pytestmark = pytest.mark.slow


def torch_detections(model, categories: List[str], image: np.ndarray) -> List[Bbox]:
    """Run the original torch model and convert its detections to normalized 'Bbox' primitives. Labels are task
    class indices (one class per model label)."""
    class_names = task_from_categories(categories).get_class_names() or []
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        output = model([image_tensor])[0]
    height, width = image.shape[:2]
    detections = []
    for (x0, y0, x1, y1), label, score in zip(
        output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist(), strict=True
    ):
        detections.append(
            Bbox(
                top_left_x=x0 / width,
                top_left_y=y0 / height,
                width=(x1 - x0) / width,
                height=(y1 - y0) / height,
                class_name=class_names[label],
                confidence=score,
            )
        )
    return detections


def score_threshold(model) -> float:
    return model.get_model_info().graph.get_component("to_bboxes").params["score_threshold"]


def test_torch_bundle_model_info(ssd_torch, frcnn_torch, micro_coco):
    ssd_info, frcnn_info = ModelBundle.from_path(ssd_torch[2]), ModelBundle.from_path(frcnn_torch[2])
    coco_class_names = micro_coco.info.get_task_by_name("object_detection").get_class_names()

    for info in (ssd_info, frcnn_info):
        # One class per model label: the 80 COCO classes plus unique placeholders for background and unused ids
        class_names = info.tasks[0].get_class_names()
        assert info.framework == "torchvision"
        assert len(class_names) == 91
        assert (class_names[0], class_names[12]) == ("NotDefined_000", "NotDefined_012")
        assert [name for name in class_names if not name.startswith("NotDefined_")] == coco_class_names
        # The network is stored by reference: weights plus the parameters to rebuild the module
        network = info.graph.get_component("network")
        assert network.files == ["network/weights.pth"]
        assert (network.params["architecture"], network.params["num_classes"]) == (info.name, 91)
        assert sorted(network.conversions) == ["onnx", "openvino", "openvino_int8"]

    # SSD is the network only, with resize and NMS as components. Faster R-CNN is the full model.
    assert [(c.name, c.type) for c in ssd_info.graph.components] == [
        ("load_image", "load_image"),
        ("resize", "resize_image"),
        ("to_tensor", "image_to_tensor"),
        ("network", "torchvision_ssd_network"),
        ("nms", "anchor_nms"),
        ("to_bboxes", "detections_to_bboxes"),
    ]
    assert [(c.name, c.type) for c in frcnn_info.graph.components] == [
        ("load_image", "load_image"),
        ("to_tensor", "image_to_tensor"),
        ("network", "torchvision_faster_rcnn"),
        ("to_bboxes", "detections_to_bboxes"),
    ]
    assert ssd_info.graph.get_component("network").batched
    assert not frcnn_info.graph.get_component("network").batched
    assert frcnn_info.graph.get_component("to_tensor").params["mean"] is None  # Normalization is in the network


def test_onnx_bundle_model_info(ssd_torch, ssd, frcnn):
    torch_info, ssd_info, frcnn_info = (
        ModelBundle.from_path(ssd_torch[2]),
        ModelBundle.from_path(ssd[2]),
        ModelBundle.from_path(frcnn[2]),
    )
    ssd_network, frcnn_network = ssd_info.graph.get_component("network"), frcnn_info.graph.get_component("network")

    for network in (ssd_network, frcnn_network):
        assert (network.type, network.files) == ("onnx_network", ["network/model.onnx"])
        assert sorted(network.conversions) == ["openvino", "openvino_int8"]  # What is left after 'onnx'
        assert network.provenance["conversion"]["options"]["opset_version"] == 17
    assert ssd_network.provenance["parent"]["type"] == "torchvision_ssd_network"
    assert NetworkParams.model_validate(ssd_network.params).input_tensors[0].shape == [None, 3, 320, 320]
    assert NetworkParams.model_validate(frcnn_network.params).input_tensors[0].shape == [3, None, None]
    assert (ssd_network.batched, frcnn_network.batched) == (True, False)

    # Only the network is converted: tasks and all other components are carried over unchanged
    assert ssd_info.tasks == torch_info.tasks
    for component in torch_info.graph.components:
        if component.name != "network":
            assert ssd_info.graph.get_component(component.name) == component

    # Nodes kept in floating point when quantizing: every pattern must match a node, or it no longer protects
    # the graph part it names
    for network, path_bundle in ((ssd_network, ssd[2]), (frcnn_network, frcnn[2])):
        graph = onnx.load_from_string(ModelBundle.from_path(path_bundle).read_file(network.files[0])).graph
        node_names = [node.name for node in graph.node]
        patterns = network.conversions["openvino_int8"].options["ignored_node_patterns"]
        assert patterns
        for pattern in patterns:
            assert any(re.fullmatch(pattern, name) for name in node_names), pattern


def test_torch_bundles_match_the_original_models(ssd_torch, frcnn_torch, images):
    """The torch network is rebuilt from the architecture name and the stored weights (Faster R-CNN with frozen
    batch norm, as the pretrained model). It runs in the same graph as the processing components."""
    ssd_model, ssd_categories, path_ssd = ssd_torch
    frcnn_model, frcnn_categories, path_frcnn = frcnn_torch
    ssd_bundle, frcnn_bundle = GraphInferenceModel.from_path(path_ssd), GraphInferenceModel.from_path(path_frcnn)
    for model in (ssd_bundle, frcnn_bundle):
        assert {type(component).engine for component in model.graph.components} == {"torch"}

    for image in images:
        image_320 = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
        reference, predicted = torch_detections(ssd_model, ssd_categories, image_320), ssd_bundle.predict(image_320)
        assert assert_detections_match(reference, predicted, min_score=0.05, min_iou=0.999, max_score_diff=1e-4) > 0

        reference, predicted = torch_detections(frcnn_model, frcnn_categories, image), frcnn_bundle.predict(image)
        settings = {"min_score": score_threshold(frcnn_bundle), "min_iou": 0.999, "max_score_diff": 1e-4}
        assert assert_detections_match(reference, predicted, **settings) > 0
        assert_detections_match(predicted, reference, **settings)

    batch_predictions = ssd_bundle.predict_batch(images)
    for image, predictions in zip(images, batch_predictions, strict=True):
        assert_detections_match(
            ssd_bundle.predict(image), predictions, min_score=0.01, min_iou=0.999, max_score_diff=1e-4
        )


def test_onnx_export_options(ssd_torch, frcnn_torch, images, tmp_path):
    """The stored ONNX options can be overridden per conversion. Without a dynamic batch axis the network takes
    one image per call, and the graph feeds a batch image by image."""
    path_bundle = (
        ModelBundle.from_path(ssd_torch[2])
        .convert("onnx", tmp_path / "static.zip", options={"dynamic_batch": False})
        .path
    )
    network = ModelBundle.from_path(path_bundle).graph.get_component("network")
    assert NetworkParams.model_validate(network.params).input_tensors[0].shape == [1, 3, 320, 320]
    assert network.provenance["conversion"]["options"] == {"opset_version": 17, "dynamic_batch": False}

    model = GraphInferenceModel.from_path(path_bundle)
    assert network_of(model).max_batch_size == 1
    assert model.predict_batch(images[:3]) == [model.predict(image) for image in images[:3]]

    with pytest.raises(ValueError, match=r"Unknown ONNX export options \['opset'\]"):
        ModelBundle.from_path(ssd_torch[2]).convert("onnx", tmp_path / "other.zip", options={"opset": 17})
    with pytest.raises(ValueError, match=r"Unknown ONNX export options \['dynamic_batch'\]"):
        ModelBundle.from_path(frcnn_torch[2]).convert("onnx", tmp_path / "other.zip", options={"dynamic_batch": True})


def test_ssd_matches_torch_without_resize(ssd, images):
    """With an image already at the model input size, resizing is an identity on both sides, so the only
    differences are numerical (ONNX Runtime vs torch, the bundle's vs the model's postprocessing)."""
    torch_model, categories, path_bundle = ssd
    model = GraphInferenceModel.from_path(path_bundle)
    for image in images:
        image_320 = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
        n_checked = assert_detections_match(
            reference=torch_detections(torch_model, categories, image_320),
            predicted=model.predict(image_320),
            min_score=0.05,
            min_iou=0.99,
            max_score_diff=1e-3,
        )
        assert n_checked > 0


def test_ssd_matches_torch_on_original_images(ssd, images):
    """Resizing in the 'resize' component (OpenCV on uint8) differs slightly from torchvision's resize of floats."""
    torch_model, categories, path_bundle = ssd
    model = GraphInferenceModel.from_path(path_bundle)
    n_checked = sum(
        assert_detections_match(
            reference=torch_detections(torch_model, categories, image),
            predicted=model.predict(image),
            min_score=0.3,
            min_iou=0.97,
            max_score_diff=0.03,
        )
        for image in images
    )
    assert n_checked >= len(images)


def test_frcnn_matches_torch(frcnn, images):
    """Same detections as the original model in both directions (nothing missing, nothing extra). Same input
    size on both sides, so differences are numerical only."""
    torch_model, categories, path_bundle = frcnn
    model = GraphInferenceModel.from_path(path_bundle)
    settings = {"min_score": score_threshold(model), "min_iou": 0.99, "max_score_diff": 1e-3}
    for image in images:
        reference, predicted = torch_detections(torch_model, categories, image), model.predict(image)
        assert assert_detections_match(reference, predicted, **settings) > 0
        assert_detections_match(predicted, reference, **settings)


@pytest.mark.parametrize("height, width", [(400, 640), (480, 640), (500, 375), (320, 320), (300, 401)])
def test_frcnn_matches_torch_across_input_sizes(frcnn, images, height, width):
    """The in-graph resize must give the same size as torch. 400 x 640 scales by exactly 0.8, which the
    float32 size computation of the traced graph got wrong (see '_resize_image_export' in the torch component)."""
    torch_model, categories, path_bundle = frcnn
    model = GraphInferenceModel.from_path(path_bundle)
    image = cv2.resize(images[0], (width, height), interpolation=cv2.INTER_LINEAR)
    reference, predicted = torch_detections(torch_model, categories, image), model.predict(image)
    settings = {"min_score": score_threshold(model), "min_iou": 0.99, "max_score_diff": 1e-3}
    assert_detections_match(reference, predicted, **settings)
    assert_detections_match(predicted, reference, **settings)


@pytest.mark.parametrize("fill_value", [0, 128, 255, None])
def test_frcnn_images_without_proposals(frcnn, fill_value):
    """Blank and noise images give no RPN proposals in the original model. The exported graph must run the
    zero-boxes path (see '_export_safe_ops' in the torch component) and return nothing, like the original."""
    torch_model, categories, path_bundle = frcnn
    if fill_value is None:
        image = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    else:
        image = np.full((300, 400, 3), fill_value, dtype=np.uint8)
    model = GraphInferenceModel.from_path(path_bundle)
    raw_outputs = model.graph.run({"image": image}, outputs=["boxes", "scores", "labels"])

    assert torch_detections(torch_model, categories, image) == []
    assert {name: output.shape for name, output in raw_outputs.items()} == {
        "boxes": (0, 4),
        "scores": (0,),
        "labels": (0,),
    }
    assert model.predict(image) == []
