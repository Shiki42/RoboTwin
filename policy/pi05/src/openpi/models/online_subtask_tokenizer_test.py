import numpy as np
import pytest

from openpi import transforms
from openpi.models import tokenizer


def test_subtask_tokenization_aligns_targets_and_teacher_forced_action_prompt():
    state = np.zeros(32, dtype=np.float32)
    task = "Put the object in the cabinet"
    subtask = "Left arm: hold object; Right arm: reach and open drawer."
    action_tokenizer = tokenizer.PaligemmaTokenizer(max_len=200)
    subtask_tokenizer = tokenizer.PaligemmaTokenizer(max_len=200)

    tokens, mask, ar_mask, loss_mask = subtask_tokenizer.tokenize_subtask(
        task,
        state,
        subtask,
    )

    assert tokens.shape == mask.shape == ar_mask.shape == loss_mask.shape == (200,)
    assert np.all(ar_mask[loss_mask])
    assert subtask_tokenizer.decode_subtask(tokens[loss_mask]) == subtask

    transform = transforms.TokenizePromptAndSubtask(
        action_tokenizer,
        subtask_tokenizer,
        "{task}\nCurrent subtask: {subtask}",
    )
    output = transform(
        {
            "prompt": task,
            "subtask": subtask,
            "state": state,
        }
    )
    expected_tokens, expected_mask = action_tokenizer.tokenize_pi05_text(
        f"{task}\nCurrent subtask: {subtask}",
        state,
    )

    assert np.array_equal(output["tokenized_action_prompt"], expected_tokens)
    assert np.array_equal(output["tokenized_action_prompt_mask"], expected_mask)
    assert np.any(output["subtask_loss_mask"])


def test_inference_transform_does_not_require_gt_subtask():
    transform = transforms.TokenizePromptAndSubtask(
        tokenizer.PaligemmaTokenizer(max_len=200),
        tokenizer.PaligemmaTokenizer(max_len=200),
        "{task}\nCurrent subtask: {subtask}",
    )

    output = transform(
        {
            "prompt": "Put the object in the cabinet",
            "state": np.zeros(32, dtype=np.float32),
        }
    )

    assert "tokenized_prompt" in output
    assert "tokenized_action_prompt" not in output
    assert "tokenized_subtask_prompt" not in output


def test_subtask_inference_prefix_has_no_teacher_targets():
    state = np.zeros(32, dtype=np.float32)
    subtask_tokenizer = tokenizer.PaligemmaTokenizer(max_len=200)

    _, _, ar_mask, loss_mask = subtask_tokenizer.tokenize_subtask(
        "Put the object in the cabinet",
        state,
    )

    assert not np.any(ar_mask)
    assert not np.any(loss_mask)


def test_teacher_forced_action_prompt_rejects_truncation():
    action_tokenizer = tokenizer.PaligemmaTokenizer(max_len=10)

    with pytest.raises(ValueError, match="Prompt token length"):
        action_tokenizer.tokenize_action_prompt(
            "Put the object in the cabinet",
            np.zeros(32, dtype=np.float32),
            "Left arm: hold object; Right arm: reach and open drawer.",
        )


@pytest.mark.parametrize(
    "subtask",
    [
        "left_arm_wait",
        "left arm wait\nright arm open",
        "Left arm: grasp object.",
        "Right arm: open drawer.",
        "Left arm: ; Right arm: open drawer.",
        "Left arm: grasp object; Right arm:",
    ],
)
def test_subtask_tokenizer_rejects_noncanonical_text(subtask):
    subtask_tokenizer = tokenizer.PaligemmaTokenizer(max_len=200)

    with pytest.raises(ValueError, match="canonical text format|arms|semantic"):
        subtask_tokenizer.tokenize_subtask(
            "Put the object in the cabinet",
            np.zeros(32, dtype=np.float32),
            subtask,
        )
