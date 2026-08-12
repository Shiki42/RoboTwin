import pytest

from openpi.training import putcab_online_subtask_config


def test_putcab_online_subtask_uses_standard_full_finetune_recipe(monkeypatch):
    revision = "a" * 64
    monkeypatch.setenv("PARALLELVLA_DATASET_REPO", "owner/putcab_parallel50")
    monkeypatch.setenv("PARALLELVLA_SUBTASK_ANNOTATION_DIR", "/annotations")
    monkeypatch.setenv("PARALLELVLA_SUBTASK_ANNOTATION_REVISION", revision)
    monkeypatch.setenv("PARALLELVLA_TRAIN_EPISODES", ",".join(str(index) for index in range(50)))
    monkeypatch.setenv("PARALLELVLA_VALIDATION_EPISODES", "50,51,52")

    config = putcab_online_subtask_config.create_config()
    data = config.data.create(config.assets_dirs, config.model)
    repack = config.data.repack_transforms.inputs[0].structure

    assert config.name == "pi05_putcab_online_subtask_pytorch_full"
    assert config.pytorch_trainable_scope == "all"
    assert config.model.casm_mode == "none"
    assert config.model.online_subtask_prediction
    assert config.model.lambda_subtask == 0.1
    assert config.model.subtask_annotation_revision == revision
    assert config.model.subtask_action_prompt_format == ("{task}\nCurrent subtask: {subtask}")
    assert config.train_episodes == tuple(range(50))
    assert config.validation_episodes == (50, 51, 52)
    assert config.validation_interval == 1_000
    assert config.batch_size == 16
    assert config.num_train_steps == 5_000
    assert config.num_workers == 4
    assert data.subtask_annotation_dir == "/annotations"
    assert data.subtask_annotation_revision == revision
    assert repack["subtask"] == "subtask"
    assert repack["action_is_pad"] == "action_is_pad"


def test_episode_index_parser_rejects_empty_value(monkeypatch):
    monkeypatch.setenv("EMPTY_EPISODES", "")

    with pytest.raises(ValueError, match="at least one episode"):
        putcab_online_subtask_config._episode_indices(  # noqa: SLF001
            "EMPTY_EPISODES",
            (0,),
        )
