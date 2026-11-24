"""'ModelBundle.convert' with made-up converters: 'dummy_network' -> target 'format_b' (type 'network_b') -> target
'format_c' (type 'network_c')."""

import json
from typing import List

import pytest
from helpers import DUMMY_NETWORK_FILE, DUMMY_TYPE, random_image, write_dummy_bundle

from inference_model_core import (
    ComponentInfo,
    ConversionInfo,
    ConversionRequest,
    ModelBundle,
    list_converters,
    register_converter,
)
from inference_model_core.convert import ANY_TYPE, unregister_converter

REQUESTS: List[ConversionRequest] = []


def convert(request: ConversionRequest) -> ComponentInfo:
    """Copies the model file under a new name and records what it was asked."""
    REQUESTS.append(request)
    source = (request.path_source_dir / request.component.files[0]).read_text()
    path_file = f"{request.component.name}/model.{request.target}"
    (request.path_output_dir / request.component.name).mkdir()
    (request.path_output_dir / path_file).write_text(json.dumps({"source": json.loads(source)}))
    conversions = {name: info for name, info in request.component.conversions.items() if name != request.target}
    converted_type = request.target.replace("format", "network")
    return request.component.model_copy(
        update={"type": converted_type, "files": [path_file], "conversions": conversions}
    )


@pytest.fixture
def converters():
    register_converter(DUMMY_TYPE, "format_b", target_type="network_b")(convert)
    register_converter("network_b", "format_c", target_type="network_c")(convert)
    REQUESTS.clear()
    yield
    unregister_converter(DUMMY_TYPE, "format_b")
    unregister_converter("network_b", "format_c")


def conversions(**infos):
    return {name: ConversionInfo(**info) for name, info in infos.items()}


def test_convert_one_component(tmp_path, converters):
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_b={"options": {"a": 1}}))
    path_bundle = (
        ModelBundle.from_path(path_source).convert("format_b", tmp_path / "converted.zip", options={"b": 2}).path
    )

    source, converted = ModelBundle.from_path(path_source).graph, ModelBundle.from_path(path_bundle).graph
    network = converted.get_component("network")
    assert (network.type, network.files, network.conversions) == ("network_b", ["network/model.format_b"], {})
    assert "source" in json.loads(ModelBundle.from_path(path_bundle).read_file("network/model.format_b"))
    # All other components are copied unchanged
    assert [c for c in converted.components if c.name != "network"] == [
        c for c in source.components if c.name != "network"
    ]
    assert REQUESTS[0].options == {"a": 1, "b": 2}  # Stored options, overridden and extended by the caller's
    assert (DUMMY_TYPE, "format_b") in list_converters()


def test_conversion_chain(tmp_path, converters):
    path_source = write_dummy_bundle(
        tmp_path / "model.zip", conversions=conversions(format_b={"options": {"a": 1}}, format_c={})
    )
    path_bundle = ModelBundle.from_path(path_source).convert("format_c", tmp_path / "converted", options={"c": 3}).path

    network = ModelBundle.from_path(path_bundle).graph.get_component("network")
    assert (network.type, network.files) == ("network_c", ["network/model.format_c"])
    assert [(request.target, request.options) for request in REQUESTS] == [
        ("format_b", {"a": 1}),
        ("format_c", {"c": 3}),
    ]
    # The second step converted the file of the first
    assert "source" in json.loads(ModelBundle.from_path(path_bundle).read_file("network/model.format_c"))["source"]
    assert not (path_bundle / DUMMY_NETWORK_FILE).exists()


def test_converter_of_any_type(tmp_path, converters):
    """The source type '*' converts every component type that has no converter of its own for the target."""
    used = []

    def convert_any(request: ConversionRequest) -> ComponentInfo:
        used.append(request.component.type)
        return convert(request)

    register_converter(ANY_TYPE, "format_b")(convert_any)
    try:
        path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_b={}, format_c={}))
        source = ModelBundle.from_path(path_source)
        assert (ANY_TYPE, "format_b") in list_converters()

        source.convert("format_b", tmp_path / "own.zip")
        assert used == []  # The converter of the component's own type comes first

        unregister_converter(DUMMY_TYPE, "format_b")
        network = source.convert("format_c", tmp_path / "chain.zip").graph.get_component("network")
        assert used == [DUMMY_TYPE]  # Also as a step of a chain
        assert network.type == "network_c"
    finally:
        unregister_converter(ANY_TYPE, "format_b")


def test_chain_needs_every_step_to_be_supported(tmp_path, converters):
    # 'format_c' is reached through 'format_b', which this component does not list
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_c={}))
    with pytest.raises(ValueError, match=r"'network' \(type 'dummy_network'\) has no converter chain"):
        ModelBundle.from_path(path_source).convert("format_c", tmp_path / "converted.zip")


def test_unsupported_conversions(tmp_path, converters):
    unsupported = conversions(format_b={"supported": False, "reason": "it has no weights"})
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=unsupported)
    with pytest.raises(ValueError, match="'network' does not support it: it has no weights"):
        ModelBundle.from_path(path_source).convert("format_b", tmp_path / "converted.zip")

    path_source = write_dummy_bundle(tmp_path / "plain.zip")
    with pytest.raises(ValueError, match=r"No component of .* can be converted to 'format_b'"):
        ModelBundle.from_path(path_source).convert("format_b", tmp_path / "converted.zip")
    with pytest.raises(ValueError, match="Unknown conversion target 'nothing'"):
        ModelBundle.from_path(path_source).convert("nothing", tmp_path / "converted.zip")


@pytest.mark.parametrize(("batched", "expected_shape"), [(True, (1, 3, 32, 32)), (False, (3, 32, 32))])
def test_calibration_feeds_the_inputs_of_the_component(tmp_path, converters, batched, expected_shape):
    # The dummy network is not registered: calibration only runs the components before the converted one
    path_source = write_dummy_bundle(tmp_path / "model.zip", batched=batched, conversions=conversions(format_b={}))
    samples = [{"image": random_image(20, 40, seed=seed)} for seed in range(3)]
    ModelBundle.from_path(path_source).convert("format_b", tmp_path / "converted.zip", calibration_inputs=samples)

    (request,) = REQUESTS
    fed = list(request.calibration())
    assert request.n_calibration_samples == len(fed) == 3
    assert all(list(inputs) == ["images"] and inputs["images"].shape == expected_shape for inputs in fed)

    REQUESTS.clear()
    ModelBundle.from_path(path_source).convert("format_b", tmp_path / "converted.zip", overwrite=True)
    assert REQUESTS[0].calibration is None


def test_converters_must_keep_the_interface(tmp_path, converters):
    def rename(request: ConversionRequest) -> ComponentInfo:
        return convert(request).model_copy(
            update={"outputs": {"boxes": "other", "scores": "scores", "labels": "labels"}}
        )

    register_converter(DUMMY_TYPE, "format_b")(rename)
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_b={}))
    with pytest.raises(ValueError, match="must keep the component's name and output keys"):
        ModelBundle.from_path(path_source).convert("format_b", tmp_path / "converted.zip")
