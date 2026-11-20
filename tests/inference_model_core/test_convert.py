"""'convert_bundle' with made-up converters: dummy -> 'format_b' -> 'format_c'."""

import json
from typing import List

import pytest
from helpers import DUMMY_NETWORK_FILE, DUMMY_RUNTIME, random_image, write_dummy_bundle

from inference_model_core import (
    ComponentInfo,
    ConversionInfo,
    ConversionRequest,
    convert_bundle,
    list_converters,
    read_bundle_file,
    read_model_info,
    register_converter,
)
from inference_model_core.convert import unregister_converter

REQUESTS: List[ConversionRequest] = []


def convert(request: ConversionRequest) -> ComponentInfo:
    """Copies the model file under a new name and records what it was asked."""
    REQUESTS.append(request)
    source = (request.path_source_dir / request.component.files[0]).read_text()
    path_file = f"{request.component.name}/model.{request.target}"
    (request.path_output_dir / request.component.name).mkdir()
    (request.path_output_dir / path_file).write_text(json.dumps({"source": json.loads(source)}))
    conversions = {name: info for name, info in request.component.conversions.items() if name != request.target}
    return request.component.model_copy(
        update={"runtime": request.target, "files": [path_file], "conversions": conversions}
    )


@pytest.fixture
def converters():
    register_converter(DUMMY_RUNTIME, "format_b", target_runtime="format_b")(convert)
    register_converter("format_b", "format_c", target_runtime="format_c")(convert)
    REQUESTS.clear()
    yield
    unregister_converter(DUMMY_RUNTIME, "format_b")
    unregister_converter("format_b", "format_c")


def conversions(**infos):
    return {name: ConversionInfo(**info) for name, info in infos.items()}


def test_convert_one_component(tmp_path, converters):
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_b={"options": {"a": 1}}))
    path_bundle = convert_bundle(path_source, tmp_path / "converted.zip", target="format_b", options={"b": 2})

    source, converted = read_model_info(path_source).graph, read_model_info(path_bundle).graph
    network = converted.get_component("network")
    assert (network.runtime, network.files, network.conversions) == ("format_b", ["network/model.format_b"], {})
    assert "source" in json.loads(read_bundle_file(path_bundle, "network/model.format_b"))
    # All other components are copied unchanged
    assert [c for c in converted.components if c.name != "network"] == [
        c for c in source.components if c.name != "network"
    ]
    assert REQUESTS[0].options == {"a": 1, "b": 2}  # Stored options, overridden and extended by the caller's
    assert ("dummy", "format_b") in list_converters()


def test_conversion_chain(tmp_path, converters):
    path_source = write_dummy_bundle(
        tmp_path / "model.zip", conversions=conversions(format_b={"options": {"a": 1}}, format_c={})
    )
    path_bundle = convert_bundle(path_source, tmp_path / "converted", target="format_c", options={"c": 3})

    network = read_model_info(path_bundle).graph.get_component("network")
    assert (network.runtime, network.files) == ("format_c", ["network/model.format_c"])
    assert [(request.target, request.options) for request in REQUESTS] == [
        ("format_b", {"a": 1}),
        ("format_c", {"c": 3}),
    ]
    # The second step converted the file of the first
    assert "source" in json.loads(read_bundle_file(path_bundle, "network/model.format_c"))["source"]
    assert not (path_bundle / DUMMY_NETWORK_FILE).exists()


def test_chain_needs_every_step_to_be_supported(tmp_path, converters):
    # 'format_c' is reached through 'format_b', which this component does not list
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_c={}))
    with pytest.raises(ValueError, match=r"'network' \(runtime 'dummy'\) has no converter chain"):
        convert_bundle(path_source, tmp_path / "converted.zip", target="format_c")


def test_unsupported_conversions(tmp_path, converters):
    unsupported = conversions(format_b={"supported": False, "reason": "it has no weights"})
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=unsupported)
    with pytest.raises(ValueError, match="'network' does not support it: it has no weights"):
        convert_bundle(path_source, tmp_path / "converted.zip", target="format_b")

    path_source = write_dummy_bundle(tmp_path / "plain.zip")
    with pytest.raises(ValueError, match=r"No component of .* can be converted to 'format_b'"):
        convert_bundle(path_source, tmp_path / "converted.zip", target="format_b")
    with pytest.raises(ValueError, match="Unknown conversion target 'nothing'"):
        convert_bundle(path_source, tmp_path / "converted.zip", target="nothing")


@pytest.mark.parametrize(("batched", "expected_shape"), [(True, (1, 3, 32, 32)), (False, (3, 32, 32))])
def test_calibration_feeds_the_inputs_of_the_component(tmp_path, converters, batched, expected_shape):
    # The dummy network is not registered: calibration only runs the components before the converted one
    path_source = write_dummy_bundle(tmp_path / "model.zip", batched=batched, conversions=conversions(format_b={}))
    samples = [{"image": random_image(20, 40, seed=seed)} for seed in range(3)]
    convert_bundle(path_source, tmp_path / "converted.zip", target="format_b", calibration_inputs=samples)

    (request,) = REQUESTS
    fed = list(request.calibration())
    assert request.n_calibration_samples == len(fed) == 3
    assert all(list(inputs) == ["images"] and inputs["images"].shape == expected_shape for inputs in fed)

    REQUESTS.clear()
    convert_bundle(path_source, tmp_path / "converted.zip", target="format_b", overwrite=True)
    assert REQUESTS[0].calibration is None


def test_converters_must_keep_the_interface(tmp_path, converters):
    def rename(request: ConversionRequest) -> ComponentInfo:
        return convert(request).model_copy(
            update={"outputs": {"boxes": "other", "scores": "scores", "labels": "labels"}}
        )

    register_converter(DUMMY_RUNTIME, "format_b")(rename)
    path_source = write_dummy_bundle(tmp_path / "model.zip", conversions=conversions(format_b={}))
    with pytest.raises(ValueError, match="must keep the component's name and output keys"):
        convert_bundle(path_source, tmp_path / "converted.zip", target="format_b")
