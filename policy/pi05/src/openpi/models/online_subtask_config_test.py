import pytest

from openpi.models import pi0_config

REVISION = "3af8b3026b544d4c7da4f56d85afdd7bca303b714f763d526f476bfc3ddff943"


def test_online_subtask_config_is_part_of_input_signature():
    config = pi0_config.Pi0Config(
        pi05=True,
        online_subtask_prediction=True,
        lambda_subtask=0.2,
        subtask_annotation_revision=REVISION,
    )

    observation, _ = config.inputs_spec(batch_size=2)

    assert observation.tokenized_action_prompt.shape == (2, 200)
    assert observation.tokenized_subtask_prompt.shape == (2, 200)
    assert observation.subtask_loss_mask.shape == (2, 200)


def test_feature_off_keeps_standard_optional_fields_empty():
    observation, _ = pi0_config.Pi0Config(pi05=True).inputs_spec()

    assert observation.tokenized_action_prompt is None
    assert observation.tokenized_subtask_prompt is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "pi05": True,
                "online_subtask_prediction": True,
                "lambda_subtask": 0.0,
                "subtask_annotation_revision": REVISION,
            },
            "lambda_subtask",
        ),
        (
            {
                "pi05": True,
                "online_subtask_prediction": True,
                "lambda_subtask": 0.2,
                "subtask_annotation_revision": "not-a-revision",
            },
            "full SHA-256",
        ),
        (
            {
                "pi05": True,
                "lambda_subtask": 0.2,
            },
            "must be zero",
        ),
        (
            {
                "pi05": True,
                "casm_mode": "visual_phase_gate",
                "online_subtask_prediction": True,
                "lambda_subtask": 0.2,
                "subtask_annotation_revision": REVISION,
            },
            "standard PI0.5 action path",
        ),
    ],
)
def test_online_subtask_config_rejects_signature_drift(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pi0_config.Pi0Config(**kwargs)
