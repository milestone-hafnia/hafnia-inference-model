import numpy as np
import pytest
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import random_image
from PIL import Image

from inference_model_core import ImageTransform, load_image_rgb
from inference_model_core.preprocessing import image_to_tensor, resize_image
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
def test_resize_transform(mode, expected_scale, expected_pad):
    resized, transform = resize_image(random_image(20, 40), mode=mode, height=32, width=32)

    assert resized.shape == (32, 32, 3)
    assert (transform.original_height, transform.original_width) == (20, 40)
    assert (transform.input_height, transform.input_width) == (32, 32)
    assert (transform.scale_x, transform.scale_y) == pytest.approx(expected_scale)
    assert (transform.pad_left, transform.pad_top) == expected_pad


def test_letterbox_padding_value():
    resized, _ = resize_image(random_image(20, 40), mode="letterbox", height=32, width=32, pad_value=114)
    # 40 -> 32 wide gives 16 rows of image, centered with 8 rows of padding above and below
    assert np.all(resized[:8] == 114)
    assert np.all(resized[24:] == 114)


@pytest.mark.parametrize("mode", ["stretch", "letterbox", "keep_ratio_pad"])
def test_boxes_map_back_to_original_image(mode):
    _, transform = resize_image(random_image(30, 60), mode=mode, height=48, width=64)

    box_normalized = np.array([[0.25, 0.1, 0.75, 0.9]])
    x = box_normalized[:, [0, 2]] * transform.original_width * transform.scale_x + transform.pad_left
    y = box_normalized[:, [1, 3]] * transform.original_height * transform.scale_y + transform.pad_top
    box_input = np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], axis=1)

    np.testing.assert_allclose(transform.input_xyxy_to_normalized(box_input), box_normalized)


def test_identity_transform():
    transform = ImageTransform.identity(height=20, width=40)
    np.testing.assert_allclose(
        transform.input_xyxy_to_normalized(np.array([[10, 5, 30, 15]])), [[0.25, 0.25, 0.75, 0.75]]
    )


def test_scale_and_normalize():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    tensor = image_to_tensor(image, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    assert tensor.shape == (3, 4, 4)
    np.testing.assert_allclose(tensor, 2.0)  # (255 / 255 - 0.5) / 0.25
    assert tensor.dtype == np.float32


def test_bgr_hwc_uint8():
    image = random_image(5, 6)
    tensor = image_to_tensor(image, color_order="BGR", layout="HWC", dtype="uint8", scale=None)
    assert tensor.shape == (5, 6, 3)
    assert tensor.dtype == np.uint8
    np.testing.assert_array_equal(tensor, image[:, :, ::-1])


def test_mean_must_match_channels():
    with pytest.raises(ValueError, match="2 values, but the image has 3 channels"):
        image_to_tensor(random_image(4, 4), mean=[0.5, 0.5], std=[0.5, 0.5])


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
    _, transform = resize_image(random_image(20, 40), mode="letterbox", height=32, width=32)

    bboxes = bboxes_from_input_xyxy(
        boxes_xyxy=np.array([[0, 8, 16, 24], [0, 0, 32, 4], [0, 8, 32, 24]]),
        scores=np.array([0.9, 0.8, 0.1]),
        class_indices=np.array([1, 0, 0]),
        task=task,
        transform=transform,
        score_threshold=0.2,
    )

    # Second box lies fully in the padding (zero area after clipping), third is below the threshold
    assert len(bboxes) == 1
    bbox = bboxes[0]
    assert (bbox.class_name, bbox.class_idx, bbox.task_name, bbox.ground_truth) == ("person", 1, "detection", False)
    assert (bbox.top_left_x, bbox.top_left_y, bbox.width, bbox.height) == pytest.approx((0.0, 0.0, 0.5, 1.0))
    assert bbox.confidence == pytest.approx(0.9)

    with pytest.raises(ValueError, match="out of range"):
        bboxes_from_input_xyxy(np.array([[0, 8, 16, 24]]), np.array([0.9]), np.array([5]), task, transform)


@pytest.mark.parametrize("size", [(224, 224), (384, 384), (600, 800)])
def test_antialias_resize_matches_torchvision(size):
    """'antialias' reproduces torchvision's tensor 'Resize' (antialias=True) on uint8 images, down and up."""
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torchvision.transforms.functional")
    image = random_image(480, 640, seed=1)
    image = np.asarray(Image.fromarray(image).resize((320, 240)).resize((640, 480)))  # Smooth, like a photo
    resized, _ = resize_image(image, mode="stretch", height=size[0], width=size[1], antialias=True)

    reference = functional.resize(torch.from_numpy(image).permute(2, 0, 1).float() / 255, list(size))
    reference = reference.permute(1, 2, 0).numpy() * 255
    assert np.abs(resized.astype(np.float32) - reference).mean() < 0.5
