import pytest
from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Bbox
from helpers import make_model_info
from pydantic import ValidationError

from inference_model_core import ComponentInfo, GraphInfo, ModelBundleInfo


def component(name="step", inputs=None, outputs=None, **kwargs) -> ComponentInfo:
    return ComponentInfo(
        name=name,
        type="dummy",
        inputs={"x": "image"} if inputs is None else inputs,
        outputs={"y": "predictions"} if outputs is None else outputs,
        **kwargs,
    )


def test_json_round_trip(tmp_path):
    model_info = make_model_info()
    path_json = tmp_path / "model_info.json"
    model_info.write_json(path_json)

    loaded = ModelBundleInfo.from_json_file(path_json)
    assert loaded == model_info
    assert loaded.tasks[0].primitive is Bbox
    assert isinstance(loaded, ModelInfo)  # Still a hafnia 'ModelInfo', so hafnia benchmark code accepts it
    assert loaded.all_files() == ["network/model.json"]


@pytest.mark.parametrize("file_path", ["../model.onnx", "/abs/model.onnx", "sub\\model.onnx", "", "a/../../b"])
def test_file_path_must_stay_inside_bundle(file_path):
    with pytest.raises(ValidationError, match="relative paths inside the bundle"):
        component(files=[file_path])


@pytest.mark.parametrize("name", ["Network", "my-step", "1st", "a.b", ""])
def test_component_names_are_identifiers(name):
    with pytest.raises(ValidationError, match="lower case identifiers"):
        component(name=name)


@pytest.mark.parametrize(
    ("components", "match"),
    [
        ([component("a"), component("a", outputs={"y": "other"})], "names must be unique"),
        ([component(files=["m.bin"]), component("b", outputs={"y": "z"}, files=["m.bin"])], "files must be unique"),
        ([component(inputs={"x": "tensor"})], r"reads \['tensor'\], which no graph input or earlier component"),
        ([component(), component("b", inputs={"x": "predictions"})], r"writes \['predictions'\], which already"),
        ([component(outputs={"y": "image"})], r"writes \['image'\], which already exist"),
        ([component(outputs={"y": "boxes"})], r"No component writes the graph outputs \['predictions'\]"),
    ],
)
def test_invalid_graphs(components, match):
    with pytest.raises(ValidationError, match=match):
        GraphInfo(inputs=["image"], outputs=["predictions"], components=components)


def test_components_for_outputs():
    graph = make_model_info().graph
    assert [c.name for c in graph.components_for(["predictions"])] == [c.name for c in graph.components]
    # The inputs of the network need neither the network nor what comes after it
    assert [c.name for c in graph.components_for(["input_tensor"])] == ["load_image", "resize", "to_tensor"]
    assert graph.components_for(["image"]) == []
    with pytest.raises(KeyError, match=r"no data keys \['nothing'\]"):
        graph.components_for(["nothing"])
    assert graph.get_component("resize").type == "resize_image"


def test_other_format_versions_are_rejected():
    data = make_model_info().model_dump()
    data["format_version"] = "2.0.0"
    with pytest.raises(ValidationError, match=r"Unsupported model info format version '2\.0\.0'"):
        ModelBundleInfo.model_validate(data)

    # A bundle of the flavor design ('runtime' instead of 'graph') gets the version error, not a list of fields
    with pytest.raises(ValidationError) as error:
        ModelBundleInfo.model_validate({"format_version": "1.0.0", "name": "old", "tasks": [], "runtime": {}})
    assert error.value.error_count() == 1
