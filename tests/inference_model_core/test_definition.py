"""Component definitions are pydantic models: fields are the configuration and the ports are declared. They hold
no implementation."""

import subprocess
import sys

import pytest
from helpers import make_bundle
from pydantic import ValidationError

from inference_model_core import ComponentDefinition, Port, Ports, TensorInfo
from inference_model_core.definitions import DetectionsToBboxes, ImageToTensor, OnnxNetwork, ResizeImage

MODEL_INFO = make_bundle()


class Scale(ComponentDefinition):
    type = "scale"
    input_ports = (Port("values", "Values to scale"), Port("offset", "Added after scaling", optional=True))
    output_ports = (Port("scaled"), Port("factor"))

    factor: float = 2.0


def test_ports_are_declared():
    assert Scale().ports() == Ports(inputs=("values",), optional_inputs=("offset",), outputs=("scaled", "factor"))
    assert ResizeImage(mode="stretch", height=8, width=8).ports().outputs == ("image", "transform")
    assert DetectionsToBboxes(task_name="a").ports() == Ports(
        inputs=("boxes", "scores", "labels"), optional_inputs=("transform", "image"), outputs=("predictions",)
    )


def test_ports_of_a_network_are_its_tensor_names():
    network = OnnxNetwork(
        input_tensors=[TensorInfo(name="images", shape=[None, 3, 32, 32])],
        output_tensors=[TensorInfo(name="boxes", shape=[None, 4]), TensorInfo(name="scores", shape=[None])],
    )
    assert network.ports() == Ports(inputs=("images",), optional_inputs=(), outputs=("boxes", "scores"))


def test_invalid_port_declarations():
    with pytest.raises(TypeError, match="port names must be unique"):

        class Duplicates(ComponentDefinition):
            type = "duplicates"
            input_ports = (Port("values"), Port("values"))

    with pytest.raises(TypeError, match="only inputs can be"):

        class OptionalOutput(ComponentDefinition):
            type = "optional_output"
            output_ports = (Port("values", optional=True),)


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
    assert (info.type, info.params) == ("scale", {"factor": 4.0})
    assert Scale.from_info(info) == Scale(factor=4)
    assert Scale.from_info(info, params={"factor": 5}).factor == 5.0  # With an override

    with pytest.raises(ValueError, match=r"Missing inputs: \['values'\], unknown inputs: \['numbers'\]"):
        Scale().connect("scale", inputs={"numbers": "a"}, outputs={"scaled": "b"})
    with pytest.raises(ValueError, match=r"unknown outputs: \['result'\]"):
        Scale().connect("scale", inputs={"values": "a"}, outputs={"result": "b"})
    with pytest.raises(ValueError, match="defines type 'resize_image'"):
        ResizeImage.from_info(info)


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


def test_detections_to_bboxes_needs_transform_or_image():
    ports = {port: port for port in ("boxes", "scores", "labels")}
    to_bboxes = DetectionsToBboxes(task_name="object_detection")
    to_bboxes.connect("to_bboxes", inputs={**ports, "transform": "t"}, outputs={"predictions": "p"})
    to_bboxes.connect("to_bboxes", inputs={**ports, "image": "i"}, outputs={"predictions": "p"})
    for inputs in (ports, {**ports, "transform": "t", "image": "i"}):
        with pytest.raises(ValueError, match="needs either the 'transform' or the 'image' input"):
            to_bboxes.connect("to_bboxes", inputs=inputs, outputs={"predictions": "p"})


def test_definitions_need_no_tensor_library():
    """Bundles can be described and checked without torch: only running a graph needs the graph engine."""
    code = (
        "import sys, inference_model_core, inference_model_core.definitions, model_exporters.definitions; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
