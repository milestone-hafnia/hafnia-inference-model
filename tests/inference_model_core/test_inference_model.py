import pytest
from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.benchmark.inference_model import InferenceModel
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import DUMMY_CLASS_NAMES, make_bundle, random_image, write_dummy_bundle
from PIL import Image

from inference_model_core import GraphInferenceModel, ModelBundle, load_graph


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


def test_params_and_write_model(tmp_path, dummy_network):
    path_bundle = write_dummy_bundle(tmp_path / "model.zip")
    model = GraphInferenceModel.from_path(path_bundle, params={"score_threshold": 0.5})
    assert [p.confidence for p in model.predict(random_image(32, 32))] == pytest.approx([0.9])

    path_copy = model.write_model(tmp_path / "copy")
    assert ModelBundle.from_path(path_copy) == ModelBundle.from_path(path_bundle)  # Overrides are not written


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
