import numpy as np
import pytest
import torch
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox
from helpers import random_image
from PIL import Image

from inference_model_core import ImageTransform
from inference_model_torch.components.detection import bboxes_from_input_xyxy, input_xyxy_to_normalized
from inference_model_torch.components.image import image_to_tensor, load_image_rgb, resize_image


def random_tensor_image(height: int, width: int, seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(random_image(height, width, seed=seed))


def photo_like_image(height: int, width: int, seed: int = 1) -> np.ndarray:
    """Random noise, smoothed like a photo: a resize of pure noise depends on every last filter detail."""
    image = Image.fromarray(random_image(height, width, seed=seed))
    return np.array(image.resize((width // 2, height // 2)).resize((width, height)))


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
    resized, transform = resize_image(random_tensor_image(20, 40), mode=mode, height=32, width=32)

    assert resized.shape == (32, 32, 3) and resized.dtype == torch.uint8
    assert (transform.original_height, transform.original_width) == (20, 40)
    assert (transform.input_height, transform.input_width) == (32, 32)
    assert (transform.scale_x, transform.scale_y) == pytest.approx(expected_scale)
    assert (transform.pad_left, transform.pad_top) == expected_pad


def test_letterbox_padding_value():
    resized, _ = resize_image(random_tensor_image(20, 40), mode="letterbox", height=32, width=32, pad_value=114)
    # 40 -> 32 wide gives 16 rows of image, centered with 8 rows of padding above and below
    assert torch.all(resized[:8] == 114)
    assert torch.all(resized[24:] == 114)


@pytest.mark.parametrize("mode", ["stretch", "letterbox", "keep_ratio_pad"])
def test_boxes_map_back_to_original_image(mode):
    _, transform = resize_image(random_tensor_image(30, 60), mode=mode, height=48, width=64)

    box_normalized = torch.tensor([[0.25, 0.1, 0.75, 0.9]], dtype=torch.float64)
    x = box_normalized[:, [0, 2]] * transform.original_width * transform.scale_x + transform.pad_left
    y = box_normalized[:, [1, 3]] * transform.original_height * transform.scale_y + transform.pad_top
    box_input = torch.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], dim=1)

    assert torch.allclose(input_xyxy_to_normalized(box_input, transform), box_normalized)


def test_identity_transform():
    transform = ImageTransform.identity(height=20, width=40)
    normalized = input_xyxy_to_normalized(torch.tensor([[10, 5, 30, 15]]), transform)
    assert normalized.tolist() == [[0.25, 0.25, 0.75, 0.75]]


@pytest.mark.parametrize("size", [(320, 320), (384, 512), (600, 800)])
def test_resize_matches_opencv(size):
    """Linear interpolation without antialiasing is OpenCV's 'INTER_LINEAR' within one gray level (OpenCV
    interpolates uint8 images in fixed point)."""
    cv2 = pytest.importorskip("cv2")
    image = photo_like_image(480, 640)
    resized, _ = resize_image(torch.from_numpy(image), mode="stretch", height=size[0], width=size[1])

    reference = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    difference = np.abs(resized.numpy().astype(np.int16) - reference)
    assert difference.max() <= 1 and difference.mean() < 0.25


@pytest.mark.parametrize("size", [(224, 224), (384, 384), (600, 800)])
def test_antialias_resize_matches_torchvision(size):
    """'antialias' reproduces torchvision's tensor 'Resize' (antialias=True) on uint8 images, down and up."""
    functional = pytest.importorskip("torchvision.transforms.functional")
    image = torch.from_numpy(photo_like_image(480, 640))
    resized, _ = resize_image(image, mode="stretch", height=size[0], width=size[1], antialias=True)

    reference = functional.resize(image.permute(2, 0, 1), list(size), antialias=True).permute(1, 2, 0)
    assert torch.equal(resized, reference)


def test_antialias_needs_linear_or_cubic_interpolation():
    with pytest.raises(ValueError, match="only supported for 'linear' and 'cubic'"):
        resize_image(
            random_tensor_image(8, 8), mode="stretch", height=4, width=4, interpolation="nearest", antialias=True
        )


def test_scale_and_normalize():
    image = torch.full((4, 4, 3), 255, dtype=torch.uint8)
    tensor = image_to_tensor(image, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    assert tensor.shape == (3, 4, 4)
    assert torch.allclose(tensor, torch.tensor(2.0))  # (255 / 255 - 0.5) / 0.25
    assert tensor.dtype == torch.float32


def test_bgr_hwc_uint8():
    image = random_tensor_image(5, 6)
    tensor = image_to_tensor(image, color_order="BGR", layout="HWC", dtype="uint8", scale=None)
    assert tensor.shape == (5, 6, 3)
    assert tensor.dtype == torch.uint8
    assert torch.equal(tensor, image.flip(-1))


def test_mean_must_match_channels():
    with pytest.raises(ValueError, match="2 values, but the image has 3 channels"):
        image_to_tensor(random_tensor_image(4, 4), mean=[0.5, 0.5], std=[0.5, 0.5])


def test_load_image_rgb_variants(tmp_path):
    image = random_image(6, 8)
    expected = torch.from_numpy(image)
    path_image = tmp_path / "image.png"
    Image.fromarray(image).save(path_image)
    read_only = image.copy()
    read_only.flags.writeable = False

    for source in (str(path_image), path_image, Image.fromarray(image), image, read_only, expected):
        loaded = load_image_rgb(source)
        assert loaded.dtype == torch.uint8 and torch.equal(loaded, expected)
    assert torch.equal(load_image_rgb(np.dstack([image, image[:, :, :1]])), expected)  # RGBA
    assert load_image_rgb(image[:, :, 0]).shape == (6, 8, 3)  # Gray scale

    with pytest.raises(ValueError, match="uint8"):
        load_image_rgb(image.astype(np.float32))
    with pytest.raises(TypeError, match="Unsupported image type 'list'"):
        load_image_rgb([[0, 0, 0]])


def test_bboxes_from_input_xyxy():
    task = TaskInfo.from_class_names(primitive=Bbox, class_names=["car", "person"], name="detection")
    _, transform = resize_image(random_tensor_image(20, 40), mode="letterbox", height=32, width=32)

    bboxes = bboxes_from_input_xyxy(
        boxes_xyxy=torch.tensor([[0, 8, 16, 24], [0, 0, 32, 4], [0, 8, 32, 24]]),
        scores=torch.tensor([0.9, 0.8, 0.1]),
        class_indices=torch.tensor([1, 0, 0]),
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
        bboxes_from_input_xyxy(torch.tensor([[0, 8, 16, 24]]), torch.tensor([0.9]), torch.tensor([5]), task, transform)
