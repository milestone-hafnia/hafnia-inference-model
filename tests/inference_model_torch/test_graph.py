import numpy as np
import pytest
import torch
from helpers import (
    DUMMY_CLASS_NAMES,
    DUMMY_DETECTIONS,
    DUMMY_NETWORK_FILE,
    DummyNetwork,
    DummyNetworkOnOtherEngine,
    make_bundle,
    random_image,
    write_dummy_bundle,
)
from pydantic import ValidationError

from inference_model_core import GraphInfo, load_graph
from inference_model_torch import TorchModelGraph, register_component
from inference_model_torch.component import unregister_component

IMAGES = [random_image(32, 32, seed=0), random_image(20, 40, seed=1), random_image(64, 16, seed=2)]


def network(graph) -> DummyNetwork:
    return next(component for component in graph.components if isinstance(component, DummyNetwork))


def test_torch_is_the_graph_engine(tmp_path, dummy_network):
    assert isinstance(load_graph(write_dummy_bundle(tmp_path / "model.zip")), TorchModelGraph)
    with pytest.raises(TypeError, match="Unknown load settings"):  # There is no graph engine to choose
        load_graph(tmp_path / "model.zip", graph_engine="torch")


def test_run_is_a_dictionary_in_and_out(tmp_path, dummy_network):
    graph = load_graph(write_dummy_bundle(tmp_path / "model.zip"))
    outputs = graph.run({"image": IMAGES[0]})

    assert list(outputs) == ["predictions"]
    assert [p.class_name for p in outputs["predictions"]] == DUMMY_CLASS_NAMES
    assert (outputs["predictions"][0].top_left_x, outputs["predictions"][0].width) == pytest.approx((0.0, 0.5))


@pytest.mark.parametrize(
    ("kwargs", "expected_input_shapes"),
    [
        ({"batched": True}, [(3, 3, 32, 32)]),  # One call with the samples stacked on a new first axis
        ({"batched": False}, [(3, 32, 32)] * 3),  # One call per sample, without a batch axis
        ({"batched": True, "max_batch_size": 2}, [(2, 3, 32, 32), (1, 3, 32, 32)]),
    ],
)
def test_run_batch(tmp_path, dummy_network, kwargs, expected_input_shapes):
    graph = load_graph(write_dummy_bundle(tmp_path / "model.zip", **kwargs))
    outputs = graph.run_batch({"image": IMAGES})

    assert network(graph).input_shapes == expected_input_shapes
    assert outputs["predictions"] == [graph.run({"image": image})["predictions"] for image in IMAGES]
    assert graph.run_batch({"image": []}) == {"predictions": []}


def test_components_exchange_torch_tensors(tmp_path, dummy_network):
    graph = load_graph(write_dummy_bundle(tmp_path / "model.zip"))
    outputs = graph.run({"image": IMAGES[1]}, outputs=["image_rgb", "input_tensor", "scores"])

    assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in outputs.values())
    assert (outputs["image_rgb"].dtype, outputs["input_tensor"].dtype) == (torch.uint8, torch.float32)
    # A tensor as graph input gives the same as the array
    as_tensor = graph.run({"image": torch.from_numpy(IMAGES[1])}, outputs=["input_tensor"])
    assert torch.equal(as_tensor["input_tensor"], outputs["input_tensor"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a CUDA device")
def test_tensors_stay_on_the_device_of_the_graph(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    graph = load_graph(path_bundle, device="CUDA")
    outputs = graph.run({"image": IMAGES[1]}, outputs=["image_rgb", "input_tensor", "scores", "predictions"])

    assert {outputs[key].device.type for key in ("image_rgb", "input_tensor", "scores")} == {"cuda"}
    assert outputs["predictions"] == load_graph(path_bundle).run({"image": IMAGES[1]})["predictions"]
    timings = {}
    graph.run_batch({"image": IMAGES}, timings=timings)
    assert set(timings) == {"load_image", "resize", "to_tensor", "network", "to_bboxes"}


def test_intermediate_outputs_only_run_what_they_need(tmp_path, dummy_network):
    graph = load_graph(write_dummy_bundle(tmp_path / "model.zip"))
    outputs = graph.run({"image": IMAGES[1]}, outputs=["input_tensor", "image_transform"])

    assert outputs["input_tensor"].shape == (3, 32, 32)
    assert outputs["image_transform"].original_width == 40
    assert network(graph).input_shapes == []  # The network comes after the requested values

    raw = graph.run({"image": IMAGES[1]}, outputs=["scores"])
    np.testing.assert_allclose(raw["scores"], DUMMY_DETECTIONS["scores"])


def test_load_for_outputs_loads_only_needed_components(tmp_path):
    # The dummy network is not registered: it is not needed for the inputs of the network
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    graph = load_graph(path_bundle, outputs=["input_tensor"])

    assert [component.info.name for component in graph.components] == ["load_image", "resize", "to_tensor"]
    assert graph.run({"image": IMAGES[0]}, outputs=["input_tensor"])["input_tensor"].shape == (3, 32, 32)
    with pytest.raises(ValueError, match=r"need components \['network', 'to_bboxes'\], which are not loaded"):
        graph.run({"image": IMAGES[0]})


def test_timings_per_component(tmp_path, dummy_network):
    graph = load_graph(write_dummy_bundle(tmp_path / "model.zip"))
    timings = {"network": 100.0}
    graph.run_batch({"image": IMAGES}, timings=timings)

    assert list(timings) == ["network", "load_image", "resize", "to_tensor", "to_bboxes"]
    assert timings["network"] > 100.0 and all(seconds >= 0 for seconds in timings.values())


def test_inputs_are_checked(tmp_path, dummy_network):
    graph = load_graph(write_dummy_bundle(tmp_path / "model.zip"))
    with pytest.raises(KeyError, match=r"Missing: \['image'\], unknown: \['picture'\]"):
        graph.run({"picture": IMAGES[0]})


def test_param_overrides(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    assert len(load_graph(path_bundle).run({"image": IMAGES[0]})["predictions"]) == 2  # Default: threshold 0

    for params in ({"score_threshold": 0.5}, {"to_bboxes.score_threshold": 0.5}):
        graph = load_graph(path_bundle, params=params)
        assert [p.confidence for p in graph.run({"image": IMAGES[0]})["predictions"]] == pytest.approx([0.9])
        # The model info, and so a written bundle, keeps the defaults
        assert graph.bundle.graph.get_component("to_bboxes").params["score_threshold"] == 0.0

    for unknown in ("unknown_param", "resize.score_threshold", "nothing.score_threshold"):
        with pytest.raises(KeyError, match="No component parameter"):
            load_graph(path_bundle, params={unknown: 1})
    with pytest.raises(ValidationError, match="score_threshold"):
        load_graph(path_bundle, params={"score_threshold": 2.0})


def test_inference_engine_override_and_options(tmp_path, dummy_network):
    register_component(alternative=True)(DummyNetworkOnOtherEngine)
    try:
        path_bundle = write_dummy_bundle(tmp_path / "model.zip")
        assert type(network(load_graph(path_bundle))) is DummyNetwork  # The default implementation

        graph = load_graph(
            path_bundle, inference_engine="other", device="GPU", num_threads=2, engine_options={"other": {"hint": 1}}
        )
        assert type(network(graph)) is DummyNetworkOnOtherEngine
        options = network(graph).context.options
        assert (options.device, options.num_threads, options.for_engine("other")) == ("GPU", 2, {"hint": 1})
        # 'GPU' is a device of another inference engine: the components written in torch use the CPU
        assert graph.run({"image": IMAGES[0]}, outputs=["input_tensor"])["input_tensor"].device.type == "cpu"
        assert graph.bundle == make_bundle()  # The bundle is unchanged
    finally:
        unregister_component(DummyNetworkOnOtherEngine)

    with pytest.raises(ValueError, match=r"No component of .* runs on inference engine 'other'"):
        load_graph(path_bundle, inference_engine="other")
    with pytest.raises(ValueError, match=r"given for engines \['other'\], which run no component"):
        load_graph(path_bundle, engine_options={"other": {}})


def test_unknown_component(tmp_path):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")  # 'dummy_network' fixture not used: not registered
    with pytest.raises(KeyError, match="no implementation of component type 'dummy_network'"):
        load_graph(path_bundle)


def test_duplicate_registration(dummy_network):
    class OtherNetwork(DummyNetwork):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_component(OtherNetwork)


def test_device_names_of_other_engines(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    # 'GPU' is no torch device: it is for the engine of the network, and the torch components use the CPU
    graph = load_graph(path_bundle, device="GPU")
    assert {component.device.type for component in graph.components} == {"cpu"}

    # Without the network every component runs on torch, so no engine would use the device
    for device in ("GPU", "cuda0"):
        with pytest.raises(ValueError, match=f"Unsupported device '{device}'"):
            load_graph(path_bundle, device=device, outputs=["input_tensor"])
    assert load_graph(path_bundle, device="CPU", outputs=["input_tensor"])


def test_batched_components_need_equal_shapes(tmp_path, dummy_network):
    # Without the resize, the images reach the batched network with different sizes
    bundle = make_bundle()
    components = {component.name: component for component in bundle.graph.components}
    to_tensor = components["to_tensor"].model_copy(update={"inputs": {"image": "image_rgb"}})
    graph_info = GraphInfo(
        inputs=["image"], outputs=["boxes"], components=[components["load_image"], to_tensor, components["network"]]
    )
    path_bundle = tmp_path / "model.zip"
    bundle.model_copy(update={"graph": graph_info}).write(path_bundle, files={DUMMY_NETWORK_FILE: b"{}"})

    with pytest.raises(ValueError, match="Can't batch input 'images' of component 'network'"):
        load_graph(path_bundle).run_batch({"image": IMAGES})
