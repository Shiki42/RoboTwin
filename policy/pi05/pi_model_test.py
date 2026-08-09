from pathlib import Path

import numpy as np
import pytest

from openpi.training.robotwin_routing import LearnedAsyncToSyncRouter
from pi_model import PI0
from pi_model import checkpoint_asset_id


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


class FakePolicy:
    def __init__(self, outputs):
        self.outputs = outputs

    def infer(self, observation):
        assert observation["prompt"] == "put the object away"
        return self.outputs


def learned_model(outputs):
    model = object.__new__(PI0)
    model.observation_window = {"prompt": "put the object away"}
    model.learned_phase_routing = True
    model.semantic_subtask_prediction = False
    model.main_camera_router = LearnedAsyncToSyncRouter(sync_confirmations=1)
    model.policy = FakePolicy(outputs)
    model.pi0_step = 50
    model.sync_action_chunk_steps = 10
    return model


def test_visual_gate_probability_drives_deployment_router():
    actions = np.ones((50, 14), dtype=np.float32)
    model = learned_model({"actions": actions, "async_probability": np.array(0.2)})

    assert model.get_action() is actions
    assert not model.main_camera_router.async_phase
    assert model.execution_steps() == 10


def test_visual_gate_deployment_requires_probability_output():
    model = learned_model({"actions": np.ones((50, 14), dtype=np.float32)})

    with pytest.raises(ValueError, match="async_probability"):
        model.get_action()


def test_casm_lan_records_exact_semantic_prompt():
    actions = np.ones((50, 14), dtype=np.float32)
    outputs = {
        "actions": actions,
        "async_probability": np.array(0.8),
        "semantic_subtask_id": 0,
        "semantic_subtask_prompt": "Left arm: grasp. Right arm: open drawer.",
        "semantic_object_arm_right_probability": 0.1,
        "semantic_stage_probabilities": [0.01, 0.02, 0.9, 0.02, 0.02, 0.03],
    }
    model = learned_model(outputs)
    model.semantic_subtask_prediction = True
    model.semantic_subtask_history = []
    model.observation_window["phase_id"] = np.array([1], dtype=np.int32)

    assert model.get_action() is actions
    assert model.semantic_subtask_history == [
        {
            "semantic_subtask_id": 0,
            "semantic_subtask_prompt": "Left arm: grasp. Right arm: open drawer.",
            "object_arm_right_probability": 0.1,
            "stage_probabilities": [0.01, 0.02, 0.9, 0.02, 0.02, 0.03],
            "confirmed_phase_id": 1,
        }
    ]


class CountingRouter:
    def __init__(self):
        self.advances = 0
        self.async_phase = False

    def advance(self):
        self.advances += 1


def test_observation_update_and_remote_advance_share_phase_semantics():
    model = object.__new__(PI0)
    model.learned_phase_routing = False
    model.main_camera_router = CountingRouter()
    model.base_instruction = "put the object away"
    images = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8)
        for index in range(3)
    ]
    state = np.arange(14, dtype=np.float32)

    model.update_observation_window(images, state, action_executed=True)

    assert model.main_camera_router.advances == 1
    assert np.array_equal(
        model.observation_window["images"]["cam_high"],
        np.transpose(images[0], (2, 0, 1)),
    )
    assert np.array_equal(model.observation_window["state"], state)

    model.advance_after_action()
    assert model.main_camera_router.advances == 2


def test_visual_phase_router_does_not_advance_between_queries():
    model = object.__new__(PI0)
    model.learned_phase_routing = True
    model.main_camera_router = CountingRouter()

    model.advance_after_action()

    assert model.main_camera_router.advances == 0
