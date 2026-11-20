import numpy as np
import pytest
from helpers import (
    DUMMY_CLASS_NAMES,
    DUMMY_DETECTIONS,
    DUMMY_NETWORK_FILE,
    DummyNetwork,
    DummyNetworkOnOtherEngine,
    make_model_info,
    random_image,
    write_dummy_bundle,
)
from pydantic import ValidationError

from inference_model_core import GraphInfo, ModelGraph, register_component, write_bundle
from inference_model_core.component import unregister_component

IMAGES = [random_image(32, 32, seed=0), random_image(20, 40, seed=1), random_image(64, 16, seed=2)]


def network(graph: ModelGraph) -> DummyNetwork:
    return next(component for component in graph.components if isinstance(component, DummyNetwork))


def test_run_is_a_dictionary_in_and_out(tmp_path, dummy_network):
    graph = ModelGraph.load(write_dummy_bundle(tmp_path / "model.zip"))
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
    graph = ModelGraph.load(write_dummy_bundle(tmp_path / "model.zip", **kwargs))
    outputs = graph.run_batch({"image": IMAGES})

    assert network(graph).input_shapes == expected_input_shapes
    assert outputs["predictions"] == [graph.run({"image": image})["predictions"] for image in IMAGES]
    assert graph.run_batch({"image": []}) == {"predictions": []}


def test_intermediate_outputs_only_run_what_they_need(tmp_path, dummy_network):
    graph = ModelGraph.load(write_dummy_bundle(tmp_path / "model.zip"))
    outputs = graph.run({"image": IMAGES[1]}, outputs=["input_tensor", "image_transform"])

    assert outputs["input_tensor"].shape == (3, 32, 32)
    assert outputs["image_transform"].original_width == 40
    assert network(graph).input_shapes == []  # The network comes after the requested values

    raw = graph.run({"image": IMAGES[1]}, outputs=["scores"])
    np.testing.assert_allclose(raw["scores"], DUMMY_DETECTIONS["scores"])


def test_load_for_outputs_loads_only_needed_components(tmp_path):
    # The dummy network is not registered: it is not needed for the inputs of the network
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    graph = ModelGraph.load(path_bundle, outputs=["input_tensor"])

    assert [component.info.name for component in graph.components] == ["load_image", "resize", "to_tensor"]
    assert graph.run({"image": IMAGES[0]}, outputs=["input_tensor"])["input_tensor"].shape == (3, 32, 32)
    with pytest.raises(ValueError, match=r"need components \['network', 'to_bboxes'\], which are not loaded"):
        graph.run({"image": IMAGES[0]})
    with pytest.raises(ValueError, match="partially loaded graph"):
        graph.write(tmp_path / "copy.zip")


def test_timings_per_component(tmp_path, dummy_network):
    graph = ModelGraph.load(write_dummy_bundle(tmp_path / "model.zip"))
    timings = {"network": 100.0}
    graph.run_batch({"image": IMAGES}, timings=timings)

    assert list(timings) == ["network", "load_image", "resize", "to_tensor", "to_bboxes"]
    assert timings["network"] > 100.0 and all(seconds >= 0 for seconds in timings.values())


def test_inputs_are_checked(tmp_path, dummy_network):
    graph = ModelGraph.load(write_dummy_bundle(tmp_path / "model.zip"))
    with pytest.raises(KeyError, match=r"Missing: \['image'\], unknown: \['picture'\]"):
        graph.run({"picture": IMAGES[0]})


def test_param_overrides(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    assert len(ModelGraph.load(path_bundle).run({"image": IMAGES[0]})["predictions"]) == 2  # Default: threshold 0

    for params in ({"score_threshold": 0.5}, {"to_bboxes.score_threshold": 0.5}):
        graph = ModelGraph.load(path_bundle, params=params)
        assert [p.confidence for p in graph.run({"image": IMAGES[0]})["predictions"]] == pytest.approx([0.9])
        # The model info, and so a written bundle, keeps the defaults
        assert graph.model_info.graph.get_component("to_bboxes").params["score_threshold"] == 0.0

    for unknown in ("unknown_param", "resize.score_threshold", "nothing.score_threshold"):
        with pytest.raises(KeyError, match="No component parameter"):
            ModelGraph.load(path_bundle, params={unknown: 1})
    with pytest.raises(ValidationError, match="score_threshold"):
        ModelGraph.load(path_bundle, params={"score_threshold": 2.0})


def test_engine_override_and_options(tmp_path, dummy_network):
    register_component(DummyNetworkOnOtherEngine)
    try:
        path_bundle = write_dummy_bundle(tmp_path / "model.zip")
        assert type(network(ModelGraph.load(path_bundle))) is DummyNetwork  # Default engine of the runtime

        graph = ModelGraph.load(
            path_bundle, inference_engine="other", device="GPU", num_threads=2, engine_options={"other": {"hint": 1}}
        )
        assert type(network(graph)) is DummyNetworkOnOtherEngine
        options = network(graph).context.options
        assert (options.device, options.num_threads, options.for_engine("other")) == ("GPU", 2, {"hint": 1})
        assert graph.model_info == make_model_info()  # The bundle is unchanged
    finally:
        unregister_component(DummyNetworkOnOtherEngine.key())

    with pytest.raises(ValueError, match=r"No component of .* runs on engine 'other'"):
        ModelGraph.load(path_bundle, inference_engine="other")
    with pytest.raises(ValueError, match=r"given for engines \['other'\], which run no component"):
        ModelGraph.load(path_bundle, engine_options={"other": {}})


def test_unknown_component(tmp_path):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")  # 'dummy_network' fixture not used: not registered
    with pytest.raises(KeyError, match="No component is registered for type 'network' on runtime 'dummy'"):
        ModelGraph.load(path_bundle)


def test_duplicate_registration(dummy_network):
    class OtherNetwork(DummyNetwork):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_component(OtherNetwork)


def test_write_round_trip(tmp_path, dummy_network):
    graph = ModelGraph.load(write_dummy_bundle(tmp_path / "model_dir"))
    path_copy = graph.write(tmp_path / "copy.zip")
    copy = ModelGraph.load(path_copy)

    assert network(copy).detections == DUMMY_DETECTIONS
    assert copy.model_info == graph.model_info


def test_batched_components_need_equal_shapes(tmp_path, dummy_network):
    # Without the resize, the images reach the batched network with different sizes
    model_info = make_model_info()
    components = {component.name: component for component in model_info.graph.components}
    to_tensor = components["to_tensor"].model_copy(update={"inputs": {"image": "image_rgb"}})
    graph_info = GraphInfo(
        inputs=["image"], outputs=["boxes"], components=[components["load_image"], to_tensor, components["network"]]
    )
    path_bundle = write_bundle(
        tmp_path / "model.zip", model_info.model_copy(update={"graph": graph_info}), {DUMMY_NETWORK_FILE: b"{}"}
    )

    with pytest.raises(ValueError, match="Can't batch input 'images' of component 'network'"):
        ModelGraph.load(path_bundle).run_batch({"image": IMAGES})
