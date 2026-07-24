from pathlib import Path

import numpy as np

import pytest

from checkpoint_assets import checkpoint_asset_id
from pi_model import EPISODE_SEED_KEY, INFERENCE_INDEX_KEY, PI0


def write_norm_stats(assets_dir: Path, asset_id: str) -> None:
    path = assets_dir / asset_id / "norm_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")


def test_checkpoint_asset_id_preserves_nested_dataset_id(tmp_path: Path):
    assets_dir = tmp_path / "assets"
    write_norm_stats(assets_dir, "Shiki42/parallelvla_putcab_clean_verified_v2_50")

    assert checkpoint_asset_id(assets_dir) == "Shiki42/parallelvla_putcab_clean_verified_v2_50"


@pytest.mark.parametrize("asset_ids", [(), ("first", "second")])
def test_checkpoint_asset_id_requires_exactly_one_norm_stats(
    tmp_path: Path,
    asset_ids: tuple[str, ...],
):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    for asset_id in asset_ids:
        write_norm_stats(assets_dir, asset_id)

    with pytest.raises(ValueError, match="exactly one"):
        checkpoint_asset_id(assets_dir)


class RecordingPolicy:
    def __init__(self):
        self.observations = []

    def infer(self, observation):
        self.observations.append(observation)
        return {"actions": np.array([[len(self.observations)]])}


def seeded_model(seed: int):
    model = object.__new__(PI0)
    model.policy = RecordingPolicy()
    model.observation_window = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 2, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 2, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 2, 3), dtype=np.uint8),
        },
        "prompt": "test instruction",
    }
    model.episode_seed = None
    model.inference_index = 0
    model._episode_action_hasher = None
    model.learned_phase_routing = False
    model.profile_inference_s = 0.0
    model.profile_inference_calls = 0
    model.audit_trace_path = None
    model.set_episode_seed(seed)
    return model


def test_episode_seed_is_forwarded_with_monotonic_inference_index():
    model = seeded_model(123)

    model.get_action()
    model.get_action()

    assert [item[EPISODE_SEED_KEY] for item in model.policy.observations] == [123, 123]
    assert [item[INFERENCE_INDEX_KEY] for item in model.policy.observations] == [0, 1]
    assert EPISODE_SEED_KEY not in model.observation_window
    assert INFERENCE_INDEX_KEY not in model.observation_window


def test_resetting_episode_seed_restarts_addressable_sequence():
    model = seeded_model(123)
    model.get_action()
    model.get_action()

    model.set_episode_seed(123)
    model.get_action()

    assert model.policy.observations[-1][INFERENCE_INDEX_KEY] == 0


def test_failed_inference_does_not_advance_index():
    model = seeded_model(123)

    def fail(_observation):
        raise RuntimeError("inference failed")

    model.policy.infer = fail
    with pytest.raises(RuntimeError, match="inference failed"):
        model.get_action()

    assert model.inference_index == 0


def test_action_digest_is_reproducible_for_identical_chunk_sequence():
    first = seeded_model(123)
    second = seeded_model(123)

    first.get_action()
    first.get_action()
    second.get_action()
    second.get_action()

    assert first.episode_action_sha256() == second.episode_action_sha256()
    assert first.inference_index == second.inference_index == 2


def learned_gate_model(probabilities):
    model = object.__new__(PI0)
    model.policy = RecordingPolicy()
    model.policy.infer = lambda observation: {
        "actions": np.zeros((50, 14), dtype=np.float32),
        "async_probability": np.asarray(probabilities.pop(0), dtype=np.float32),
    }
    model.observation_window = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 2, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 2, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 2, 3), dtype=np.uint8),
        },
        "prompt": "test instruction",
    }
    model.episode_seed = None
    model.inference_index = 0
    model._episode_action_hasher = None
    model.learned_phase_routing = True
    model.profile_inference_s = 0.0
    model.profile_inference_calls = 0
    model.audit_trace_path = None
    model.pi0_step = 50
    model.sync_action_chunk_steps = 10
    from openpi.training.robotwin_routing import LearnedAsyncToSyncRouter
    model.main_camera_router = LearnedAsyncToSyncRouter(sync_confirmations=2)
    return model


def test_visual_gate_changes_chunk_length_after_confirmed_transition():
    model = learned_gate_model([0.4, 0.4])

    model.get_action()
    assert model.execution_steps() == 50
    model.get_action()

    assert model.main_camera_router.async_phase is False
    assert model.execution_steps() == 10


def test_visual_gate_uses_dynamic_main_camera_during_async_phase():
    model = learned_gate_model([0.9])
    model.instruction = "put the object in the cabinet"
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.full((2, 3, 3), 7, dtype=np.uint8)
    wrists = np.zeros((2, 3, 3), dtype=np.uint8)
    state = np.zeros(14, dtype=np.float32)

    model.update_observation_window([first, wrists, wrists], state)
    model.update_observation_window([second, wrists, wrists], state)

    np.testing.assert_array_equal(
        model.observation_window["images"]["cam_high"],
        np.transpose(second, (2, 0, 1)),
    )


def test_local_websocket_client_disables_keepalive():
    source = (
        Path(__file__).parent
        / "packages"
        / "openpi-client"
        / "src"
        / "openpi_client"
        / "websocket_client_policy.py"
    ).read_text(encoding="utf-8")

    assert "ping_interval=None" in source
