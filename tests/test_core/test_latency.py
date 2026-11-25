import pytest
from helpers import network_of, random_image, write_dummy_bundle

from hafnia_inference_model.core import GraphInferenceModel
from hafnia_inference_model.core.inference_model import LATENCY_PERCENTILES


@pytest.fixture
def images():
    return [random_image(48, 64, seed=seed) for seed in range(5)]


def test_report_has_all_stages_and_statistics(tmp_path, images, dummy_network):
    model = GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "dummy.zip"))
    report = model.measure_latency(images, warmup=2)

    # One stage per component of the graph, and the total
    stages = ["load_image", "resize", "to_tensor", "network", "to_bboxes", "total"]
    expected = {f"{stage}_mean_ms" for stage in stages}
    expected |= {f"{stage}_p{percentile}_ms" for stage in stages for percentile in LATENCY_PERCENTILES}
    assert set(report) == expected | {"throughput_images_per_s"}
    assert all(value >= 0 for value in report.values())
    assert report["total_p99_ms"] >= report["total_p50_ms"]
    assert report["total_mean_ms"] >= sum(report[f"{stage}_mean_ms"] for stage in stages[:-1])
    assert report["throughput_images_per_s"] > 0


@pytest.mark.parametrize(
    ("batched", "n_warmup_calls", "expected_input_shapes"),
    [
        (True, 1, [(2, 3, 32, 32), (2, 3, 32, 32), (1, 3, 32, 32)]),
        (False, 2, [(3, 32, 32)] * 5),
    ],
)
def test_batches_and_warmup(tmp_path, images, dummy_network, batched, n_warmup_calls, expected_input_shapes):
    """Batched networks get one call per batch, others one per image. Warm-up calls are not timed."""
    model = GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "dummy.zip", batched=batched))
    model.measure_latency(images, batch_size=2, warmup=1)
    # The warm-up is the first batch of 2 images: 1 batched call or 2 single calls
    assert network_of(model).input_shapes[n_warmup_calls:] == expected_input_shapes


def test_invalid_arguments(tmp_path, images, dummy_network):
    model = GraphInferenceModel.from_path(write_dummy_bundle(tmp_path / "dummy.zip"))
    with pytest.raises(ValueError, match="batch_size"):
        model.measure_latency(images, batch_size=0)
    with pytest.raises(ValueError, match="At least one image"):
        model.measure_latency([])
