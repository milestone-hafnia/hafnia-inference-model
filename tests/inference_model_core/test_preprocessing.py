import numpy as np
import pytest
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import make_model_info, random_image
from PIL import Image

from inference_model_core import PreprocessingInfo, ResizeInfo, TensorInfo, load_image_rgb, preprocess_image
from inference_model_core.preprocessing import preprocess_model_input
from inference_model_core.primitives import bboxes_from_input_xyxy


@pytest.mark.parametrize(
    ("mode", "expected_scale", "expected_pad"),
    [
        # Image (H=20, W=40) into a 32 x 32 input
        ("stretch", (32 / 40, 32 / 20), (0, 0)),
        ("letterbox", (0.8, 0.8), (0, 8)),
        ("keep_ratio_pad", (0.8, 0.8), (0, 0)),
    ],
)
def test_resize_meta(mode, expected_scale, expected_pad):
    preprocessing = PreprocessingInfo(resize=ResizeInfo(mode=mode, height=32, width=32))
    tensor, meta = preprocess_image(random_image(20, 40), preprocessing)

    assert tensor.shape == (1, 3, 32, 32)
    assert (meta.original_height, meta.original_width) == (20, 40)
    assert (meta.input_height, meta.input_width) == (32, 32)
    assert (meta.scale_x, meta.scale_y) == pytest.approx(expected_scale)
    assert (meta.pad_left, meta.pad_top) == expected_pad


def test_letterbox_padding_value():
    preprocessing = PreprocessingInfo(resize=ResizeInfo(mode="letterbox", pad_value=114), scale=None)
    tensor, _ = preprocess_image(random_image(20, 40), preprocessing, input_hw=(32, 32), add_batch_dim=False)
    # 40 -> 32 wide gives 16 rows of image, centered with 8 rows of padding above and below
    assert np.all(tensor[:, :8] == 114)
    assert np.all(tensor[:, 24:] == 114)


@pytest.mark.parametrize("mode", ["stretch", "letterbox", "keep_ratio_pad"])
def test_boxes_map_back_to_original_image(mode):
    preprocessing = PreprocessingInfo(resize=ResizeInfo(mode=mode))
    _, meta = preprocess_image(random_image(30, 60), preprocessing, input_hw=(48, 64))

    box_normalized = np.array([[0.25, 0.1, 0.75, 0.9]])
    x = box_normalized[:, [0, 2]] * meta.original_width * meta.scale_x + meta.pad_left
    y = box_normalized[:, [1, 3]] * meta.original_height * meta.scale_y + meta.pad_top
    box_input = np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], axis=1)

    np.testing.assert_allclose(meta.input_xyxy_to_normalized(box_input), box_normalized)


def test_scale_and_normalize():
    preprocessing = PreprocessingInfo(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    tensor, _ = preprocess_image(image, preprocessing)
    np.testing.assert_allclose(tensor, 2.0)  # (255 / 255 - 0.5) / 0.25
    assert tensor.dtype == np.float32


def test_steps_in_graph_are_skipped():
    preprocessing = PreprocessingInfo(
        resize=ResizeInfo(mode="stretch", height=8, width=8),
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        in_graph=["resize", "scale", "normalize"],
    )
    image = random_image(20, 40)
    tensor, meta = preprocess_image(image, preprocessing, add_batch_dim=False)

    np.testing.assert_array_equal(tensor, image.transpose(2, 0, 1).astype(np.float32))
    assert (meta.input_height, meta.input_width, meta.scale_x, meta.scale_y) == (20, 40, 1.0, 1.0)


def test_bgr_nhwc_uint8():
    preprocessing = PreprocessingInfo(color_order="BGR", layout="NHWC", dtype="uint8", scale=None)
    image = random_image(5, 6)
    tensor, _ = preprocess_image(image, preprocessing)
    assert tensor.shape == (1, 5, 6, 3)
    assert tensor.dtype == np.uint8
    np.testing.assert_array_equal(tensor[0], image[:, :, ::-1])


def test_missing_resize_size():
    preprocessing = PreprocessingInfo(resize=ResizeInfo(mode="stretch"))
    with pytest.raises(ValueError, match="Resize target size is unknown"):
        preprocess_image(random_image(4, 4), preprocessing, input_hw=(None, None))


def test_load_image_rgb_variants(tmp_path):
    image = random_image(6, 8)
    path_image = tmp_path / "image.png"
    Image.fromarray(image).save(path_image)

    np.testing.assert_array_equal(load_image_rgb(str(path_image)), image)
    np.testing.assert_array_equal(load_image_rgb(Image.fromarray(image)), image)
    np.testing.assert_array_equal(load_image_rgb(np.dstack([image, image[:, :, :1]])), image)  # RGBA
    assert load_image_rgb(image[:, :, 0]).shape == (6, 8, 3)  # Gray scale

    with pytest.raises(ValueError, match="uint8"):
        load_image_rgb(image.astype(np.float32))


def test_bboxes_from_input_xyxy():
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=["car", "person"], name="detection")
    preprocessing = PreprocessingInfo(resize=ResizeInfo(mode="letterbox"))
    _, meta = preprocess_image(random_image(20, 40), preprocessing, input_hw=(32, 32))

    bboxes = bboxes_from_input_xyxy(
        boxes_xyxy=np.array([[0, 8, 16, 24], [0, 0, 32, 4], [0, 8, 32, 24]]),
        scores=np.array([0.9, 0.8, 0.1]),
        class_indices=np.array([1, 0, 0]),
        task=task,
        meta=meta,
        score_threshold=0.2,
    )

    # Second box lies fully in the padding (zero area after clipping), third is below the threshold
    assert len(bboxes) == 1
    bbox = bboxes[0]
    assert (bbox.class_name, bbox.class_idx, bbox.task_name, bbox.ground_truth) == ("person", 1, "detection", False)
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.0, 0.5, 1.0))
    assert bbox.confidence == pytest.approx(0.9)

    with pytest.raises(ValueError, match="out of range"):
        bboxes_from_input_xyxy(np.array([[0, 8, 16, 24]]), np.array([0.9]), np.array([5]), task, meta)


@pytest.mark.parametrize(
    ("shape", "layout", "in_graph", "expected_shape"),
    [
        ([None, 3, 32, 48], "NCHW", [], (1, 3, 32, 48)),  # Batched, resized to the static input size
        ([1, 32, 48, 3], "NHWC", [], (1, 32, 48, 3)),
        ([3, None, None], "NCHW", ["resize"], (3, 20, 40)),  # Rank 3 without batch axis, resize in graph
    ],
)
def test_preprocess_model_input(shape, layout, in_graph, expected_shape):
    runtime = make_model_info().runtime.model_copy(deep=True)
    runtime.input.tensors[0] = TensorInfo(name="pixels", shape=shape)
    runtime.preprocessing = PreprocessingInfo(resize=ResizeInfo(mode="stretch"), layout=layout, in_graph=in_graph)

    inputs, meta = preprocess_model_input(random_image(20, 40), runtime)

    assert list(inputs) == ["pixels"]
    assert inputs["pixels"].shape == expected_shape
    assert (meta.original_height, meta.original_width) == (20, 40)
