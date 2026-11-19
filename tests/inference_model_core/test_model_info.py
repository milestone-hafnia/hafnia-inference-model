import pytest
from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Bbox
from helpers import make_model_info
from pydantic import ValidationError

from inference_model_core import FlavorKey, ModelBundleInfo, PreprocessingInfo


def test_json_round_trip(tmp_path):
    model_info = make_model_info()
    path_json = tmp_path / "model_info.json"
    model_info.write_json(path_json)

    loaded = ModelBundleInfo.from_json_file(path_json)
    assert loaded == model_info
    assert loaded.tasks[0].primitive is Bbox
    assert isinstance(loaded, ModelInfo)  # Still a hafnia 'ModelInfo', so hafnia benchmark code accepts it


def test_flavor_key():
    assert make_model_info().flavor_key() == FlavorKey("onnxruntime", "dummy", "dummy_detection")


@pytest.mark.parametrize("file_path", ["../model.onnx", "/abs/model.onnx", "sub\\model.onnx", "", "a/../../b"])
def test_file_path_must_stay_inside_bundle(file_path):
    with pytest.raises(ValidationError, match="relative paths inside the bundle"):
        make_model_info(file_path=file_path)


def test_file_paths_must_be_unique():
    data = make_model_info().model_dump()
    data["runtime"]["extra_files"] = ["model.json"]
    with pytest.raises(ValidationError, match="unique"):
        ModelBundleInfo.model_validate(data)


def test_unsupported_format_version():
    data = make_model_info().model_dump()
    data["format_version"] = "2.0.0"
    with pytest.raises(ValidationError, match="Unsupported model info format version"):
        ModelBundleInfo.model_validate(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mean": [0.5, 0.5, 0.5]}, "both be set"),
        ({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5]}, "same length"),
        ({"mean": [0.5], "std": [0.0]}, "zeros"),
        ({"dtype": "uint8", "scale": 1 / 255}, "can not be scaled"),
    ],
)
def test_invalid_preprocessing(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        PreprocessingInfo(**kwargs)


def test_uint8_input_has_no_default_scale():
    preprocessing = PreprocessingInfo(dtype="uint8")
    assert preprocessing.scale is None
    assert PreprocessingInfo().scale == pytest.approx(1 / 255)
