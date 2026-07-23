from types import SimpleNamespace

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


def test_preprocess_return_is_fullgraph_traceable():
    image = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    observation = SimpleNamespace(
        images={"camera": image},
        image_masks={},
        state=torch.zeros(2, 7),
        tokenized_prompt=torch.ones(2, 4, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    def preprocess(input_observation):
        return preprocessing_pytorch.preprocess_observation_pytorch(
            input_observation,
            train=False,
            image_keys=("camera",),
            image_resolution=(4, 5),
        )

    compiled_preprocess = torch.compile(
        preprocess,
        backend="eager",
        fullgraph=True,
    )
    (
        images,
        image_masks,
        tokenized_prompt,
        tokenized_prompt_mask,
        state,
    ) = compiled_preprocess(observation)

    assert len(images) == 1
    assert len(image_masks) == 1
    torch.testing.assert_close(images[0], image)
    torch.testing.assert_close(image_masks[0], torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(tokenized_prompt, observation.tokenized_prompt)
    torch.testing.assert_close(tokenized_prompt_mask, observation.tokenized_prompt_mask)
    torch.testing.assert_close(state, observation.state)
