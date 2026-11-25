import sys

import pytest
from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.benchmark.inference_model import InferenceModel
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import DUMMY_CLASS_NAMES, make_bundle, random_image, write_dummy_bundle
from PIL import Image

from hafnia_inference_model.core import GraphInferenceModel, LoadSettings, ModelBundle, load_graph


def test_graph_inference_model_from_path(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    model = GraphInferenceModel.from_path(path_bundle)

    assert isinstance(model, InferenceModel)  # One class for all bundles
    assert model.get_model_info() == make_bundle()


def test_graph_inference_model_from_bundle(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    bundle = ModelBundle.from_path(path_bundle)
    model = GraphInferenceModel.from_bundle(bundle, params={"score_threshold": 0.5})

    assert model.get_model_info() is bundle  # The bundle is not read again
    image = random_image(32, 32)
    assert model.predict(image) == GraphInferenceModel.from_path(path_bundle, params={"score_threshold": 0.5}).predict(
        image
    )


def test_load_settings(tmp_path, dummy_network):
    """'from_path', 'from_bundle' and 'load_graph' take the same settings, see 'LoadSettings'."""
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    image = random_image(32, 32)
    expected = GraphInferenceModel.from_path(path_bundle).predict(image)

    # A setting that is 'None' has its default
    defaults = dict.fromkeys(LoadSettings.__annotations__)
    assert GraphInferenceModel.from_path(path_bundle, **defaults).predict(image) == expected
    assert GraphInferenceModel.from_bundle(ModelBundle.from_path(path_bundle), **defaults).predict(image) == expected

    for load in (GraphInferenceModel.from_path, load_graph):
        with pytest.raises(TypeError, match=r"Unknown load settings \['devise'\]. Settings: \['device', "):
            load(path_bundle, devise="CPU")


def test_running_a_graph_needs_the_torch_graph_engine(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hafnia_inference_model.torch.graph", None)  # As when torch is not installed
    with pytest.raises(ImportError, match=r"needs the torch graph engine .* Install the 'torch' extra"):
        load_graph(write_dummy_bundle(tmp_path / "model.zip"))


def test_from_bundle_needs_the_files_on_disk(dummy_network):
    with pytest.raises(ValueError, match="is only in memory and has no files on disk"):
        GraphInferenceModel.from_bundle(make_bundle())


def test_predict_returns_primitives(tmp_path, dummy_network):
    model = GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "model.zip"))
    predictions = model.predict(random_image(32, 32))

    assert [p.class_name for p in predictions] == DUMMY_CLASS_NAMES
    assert all(isinstance(p, Bbox) and not p.ground_truth for p in predictions)
    assert (predictions[0].top_left_x, predictions[0].width) == pytest.approx((0.0, 0.5))


def test_predict_batch(tmp_path, dummy_network):
    model = GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "model.zip"))
    images = [random_image(32, 32, seed=0), random_image(20, 40, seed=1), random_image(64, 16, seed=2)]

    assert model.predict_batch(images) == [model.predict(image) for image in images]
    with pytest.raises(ValueError, match="must match"):
        model.predict_batch([random_image(8, 8)], sample_dicts=[{}, {}])


def test_params_are_not_part_of_the_bundle(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    model = GraphInferenceModel.from_path(path_bundle, params={"score_threshold": 0.5})
    assert [p.confidence for p in model.predict(random_image(32, 32))] == pytest.approx([0.9])

    # A loaded model is written again through its bundle: overrides only apply to the loaded model
    copy = model.get_model_info().write(tmp_path / "copy")
    assert ModelBundle.from_path(copy.path) == ModelBundle.from_path(path_bundle)
    assert [p.confidence for p in GraphInferenceModel.from_bundle(copy).predict(random_image(32, 32))] != [0.9]


def test_graph_needs_image_input_and_predictions_output(tmp_path, dummy_network):
    bundle = make_bundle()
    graph_info = bundle.graph.model_copy(update={"outputs": ["boxes"]})
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")

    graph = load_graph(path_bundle)
    graph.bundle = bundle.model_copy(update={"graph": graph_info})

    with pytest.raises(ValueError, match="needs a graph with the input 'image' and an output 'predictions'"):
        GraphInferenceModel(graph)


def test_works_with_hafnia_benchmark(tmp_path, dummy_network):
    """The inference model plugs directly into hafnia's 'run_inference_on_dataset'."""
    samples = []
    for i_sample in range(3):
        path_image = tmp_path / f"image_{i_sample}.png"
        Image.fromarray(random_image(24, 48, seed=i_sample)).save(path_image)
        samples.append(Sample(file_path=str(path_image), height=24, width=48, split="test", bboxes=[]))
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=DUMMY_CLASS_NAMES, name="object_detection")
    dataset = HafniaDataset.from_samples_list(samples, info=DatasetInfo(dataset_name="dummy", tasks=[task]))

    model = GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "model.zip"))
    dataset_predictions = run_inference_on_dataset(dataset=dataset, model=model, batch_size=2)

    prediction_task = dataset_predictions.info.get_task_by_name("object_detection/predictions")
    assert prediction_task.get_class_names() == DUMMY_CLASS_NAMES
    n_predictions = dataset_predictions.samples["bboxes"].list.len().sum()
    assert n_predictions == len(samples) * len(DUMMY_CLASS_NAMES)
