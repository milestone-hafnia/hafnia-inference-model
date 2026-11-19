import numpy as np
import pytest
from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import (
    DUMMY_CLASS_NAMES,
    DUMMY_DETECTIONS,
    DummyDetector,
    make_model_info,
    random_image,
    write_dummy_bundle,
)
from PIL import Image
from pydantic import ValidationError

from inference_model_core import get_flavor, inference_model_from_path, list_flavors, register_flavor
from inference_model_core.registry import unregister_flavor


def test_inference_model_from_path(tmp_path, dummy_flavor):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    model = inference_model_from_path(path_bundle)

    assert isinstance(model, DummyDetector)
    assert model.get_model_info() == make_model_info()
    assert dummy_flavor.flavor_key() in list_flavors()


def test_unknown_flavor(tmp_path):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")  # 'dummy_flavor' fixture not used: not registered
    with pytest.raises(KeyError, match="No inference model flavor is registered"):
        inference_model_from_path(path_bundle)


def test_duplicate_flavor_key(dummy_flavor):
    class OtherDetector(DummyDetector):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_flavor(OtherDetector)
    assert get_flavor(dummy_flavor.flavor_key()) is dummy_flavor


def test_framework_and_decoder_must_match_model_info():
    model_info = make_model_info().model_copy(update={"framework": "other"})
    with pytest.raises(ValueError, match="handles framework 'dummy'"):
        DummyDetector(model_info=model_info, detections=DUMMY_DETECTIONS)


class DummyOpenVINODetector(DummyDetector):
    """Same model family on another engine, which can also load ONNX bundles."""

    inference_engine = "openvino"
    supported_formats = ("onnx", "openvino")


class DummyOpenVINOOnlyIRDetector(DummyDetector):
    inference_engine = "openvino"
    supported_formats = ("openvino",)


@pytest.fixture
def register_temporarily():
    registered = []

    def _register(flavor_class):
        register_flavor(flavor_class)
        registered.append(flavor_class)

    yield _register
    for flavor_class in registered:
        unregister_flavor(flavor_class.flavor_key())


def test_engine_override(tmp_path, dummy_flavor, register_temporarily):
    register_temporarily(DummyOpenVINODetector)
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")

    assert type(inference_model_from_path(path_bundle)) is DummyDetector  # Default engine of the bundle
    model = inference_model_from_path(path_bundle, inference_engine="openvino")
    assert type(model) is DummyOpenVINODetector
    assert model.get_model_info().runtime.inference_engine == "onnxruntime"  # The bundle is unchanged


def test_engine_override_requires_supported_format(tmp_path, register_temporarily):
    register_temporarily(DummyOpenVINOOnlyIRDetector)
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    with pytest.raises(ValueError, match="can not run model format 'onnx'"):
        inference_model_from_path(path_bundle, inference_engine="openvino")


def test_postprocessing_params_override(tmp_path, dummy_flavor):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    image = random_image(32, 32)

    assert len(inference_model_from_path(path_bundle).predict(image)) == 2  # Bundle default: threshold 0.0
    model = inference_model_from_path(path_bundle, postprocessing_params={"score_threshold": 0.5})
    assert model.params.score_threshold == 0.5
    assert [p.confidence for p in model.predict(image)] == pytest.approx([0.9])
    assert model.get_model_info().runtime.postprocessing.params == {"score_threshold": 0.0}  # Defaults unchanged

    with pytest.raises(ValidationError, match="unknown_param"):
        inference_model_from_path(path_bundle, postprocessing_params={"unknown_param": 1})


def test_write_model_round_trip(tmp_path):
    path_bundle = write_dummy_bundle(tmp_path / "model_dir")
    model = DummyDetector.load_model(path_bundle)
    path_copy = model.write_model(tmp_path / "copy.zip")
    assert DummyDetector.load_model(path_copy).detections == DUMMY_DETECTIONS


def test_predict_returns_primitives():
    model = DummyDetector(model_info=make_model_info(), detections=DUMMY_DETECTIONS)
    predictions = model.predict(random_image(32, 32))

    assert [p.class_name for p in predictions] == DUMMY_CLASS_NAMES
    assert all(isinstance(p, Bbox) and not p.ground_truth for p in predictions)
    assert (predictions[0].top_left_x, predictions[0].width) == pytest.approx((0.0, 0.5))


@pytest.mark.parametrize(("batch_dim", "expected_raw_batch_sizes"), [(None, [3]), (1, [1, 1, 1])])
def test_predict_batch(batch_dim, expected_raw_batch_sizes):
    model = DummyDetector(model_info=make_model_info(batch_dim=batch_dim), detections=DUMMY_DETECTIONS)
    images = [random_image(32, 32, seed=0), random_image(20, 40, seed=1), random_image(64, 16, seed=2)]

    batch_predictions = model.predict_batch(images)

    # Dynamic batch dimension -> one batched call, static batch of 1 -> one call per image
    assert model.raw_batch_sizes == expected_raw_batch_sizes
    assert batch_predictions == [model.predict(image) for image in images]


def test_predict_batch_sample_dicts_length():
    model = DummyDetector(model_info=make_model_info(), detections=DUMMY_DETECTIONS)
    with pytest.raises(ValueError, match="must match"):
        model.predict_batch([random_image(8, 8)], sample_dicts=[{}, {}])


def test_split_raw_outputs_requires_batch_major_outputs():
    model = DummyDetector(model_info=make_model_info(), detections=DUMMY_DETECTIONS)
    with pytest.raises(ValueError, match="not batch-major"):
        model.split_raw_outputs({"boxes": np.zeros((5, 4))}, batch_size=2)


def test_works_with_hafnia_benchmark(tmp_path):
    """Flavors plug directly into hafnia's 'run_inference_on_dataset'."""
    samples = []
    for i_sample in range(3):
        path_image = tmp_path / f"image_{i_sample}.png"
        Image.fromarray(random_image(24, 48, seed=i_sample)).save(path_image)
        samples.append(Sample(file_path=str(path_image), height=24, width=48, split="test", bboxes=[]))
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=DUMMY_CLASS_NAMES, name="object_detection")
    dataset = HafniaDataset.from_samples_list(samples, info=DatasetInfo(dataset_name="dummy", tasks=[task]))

    model = DummyDetector(model_info=make_model_info(), detections=DUMMY_DETECTIONS)
    dataset_predictions = run_inference_on_dataset(dataset=dataset, model=model, batch_size=2)

    prediction_task = dataset_predictions.info.get_task_by_name("object_detection/predictions")
    assert prediction_task.get_class_names() == DUMMY_CLASS_NAMES
    n_predictions = dataset_predictions.samples["bboxes"].list.len().sum()
    assert n_predictions == len(samples) * len(DUMMY_CLASS_NAMES)
