import torch

from openpi.models_pytorch import preprocessing_pytorch


def test_crop_by_tensor_offsets_matches_dynamic_slice():
    image = torch.arange(2 * 5 * 7 * 3).reshape(2, 5, 7, 3)
    start_h = torch.tensor([1])
    start_w = torch.tensor([2])

    compiled_crop = torch.compile(
        preprocessing_pytorch._crop_by_tensor_offsets,  # noqa: SLF001
        backend="eager",
        fullgraph=True,
    )
    cropped = compiled_crop(image, start_h, start_w, 3, 4)

    expected = image[:, 1:4, 2:6, :]
    torch.testing.assert_close(cropped, expected)


def test_rotation_threshold_is_traceable_and_preserves_skipped_rotation():
    image = torch.linspace(-1, 1, 2 * 5 * 7 * 3).reshape(2, 5, 7, 3)
    angle = torch.tensor([0.05])

    compiled_rotate = torch.compile(
        preprocessing_pytorch._rotate_if_above_threshold,  # noqa: SLF001
        backend="eager",
        fullgraph=True,
    )
    rotated = compiled_rotate(image, angle, 5, 7)

    torch.testing.assert_close(rotated, image)
