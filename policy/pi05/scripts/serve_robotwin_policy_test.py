from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_robotwin_policy as server_module


class _Router:
    def __init__(self):
        self.step = 0

    def reset(self):
        self.step = 0


class _Metrics:
    def __init__(self):
        self.chunks = []

    def reset(self):
        self.chunks = []


class _FakeModel:
    def __init__(self):
        self.policy = object()
        self.main_camera_router = _Router()
        self.activity_metrics = _Metrics()
        self.base_instruction = None
        self.observation_window = None
        self.semantic_subtask_history = []

    def reset_obsrvationwindows(self):
        self.base_instruction = None
        self.observation_window = None
        self.main_camera_router.reset()
        self.activity_metrics.reset()

    def set_episode_seed(self, seed):
        torch.manual_seed(seed)

    def set_language(self, instruction):
        self.base_instruction = instruction

    def update_observation_window(self, images, state, *, action_executed):
        del images, action_executed
        self.observation_window = copy.deepcopy(state)

    def get_action(self):
        self.main_camera_router.step += 1
        return torch.rand((2, 3)).numpy()

    def execution_steps(self):
        return 2

    def record_chunk(self, length):
        self.activity_metrics.chunks.append(length)

    def record_action(self, action, state):
        del action, state

    def advance_after_action(self):
        self.main_camera_router.step += 1

    def rollout_metrics(self):
        return {
            "executed_chunk_lengths": self.activity_metrics.chunks.copy(),
            "router_step": self.main_camera_router.step,
        }


def _request(command, **kwargs):
    return {"command": command, **kwargs}


def _infer_request():
    return _request("infer", images=[], state=[0.0], instruction="put object")


def test_interleaved_sessions_keep_rng_trace_and_router_state_isolated(monkeypatch):
    monkeypatch.setattr(server_module, "decode_images", lambda images: images)
    service = server_module.RobotwinPolicyService(_FakeModel())

    for session_id, seed in ((11, 101), (22, 202)):
        service.infer_session(session_id, _request("reset"))
        service.infer_session(session_id, _request("seed", seed=seed))

    action_11_first = service.infer_session(11, _infer_request())["actions"]
    action_22_first = service.infer_session(22, _infer_request())["actions"]
    action_11_second = service.infer_session(11, _infer_request())["actions"]

    expected = server_module.RobotwinPolicyService(_FakeModel())
    expected.infer_session(1, _request("reset"))
    expected.infer_session(1, _request("seed", seed=101))
    expected_11_first = expected.infer_session(1, _infer_request())["actions"]
    expected_11_second = expected.infer_session(1, _infer_request())["actions"]
    expected.infer_session(2, _request("reset"))
    expected.infer_session(2, _request("seed", seed=202))
    expected_22_first = expected.infer_session(2, _infer_request())["actions"]

    np.testing.assert_array_equal(action_11_first, expected_11_first)
    np.testing.assert_array_equal(action_11_second, expected_11_second)
    np.testing.assert_array_equal(action_22_first, expected_22_first)

    metrics_11 = service.infer_session(11, _request("metrics"))["metrics"]
    metrics_22 = service.infer_session(22, _request("metrics"))["metrics"]
    assert metrics_11["server_policy_inference_requests"] == 2
    assert metrics_22["server_policy_inference_requests"] == 1
    assert metrics_11["executed_chunk_lengths"] == [2, 2]
    assert metrics_22["executed_chunk_lengths"] == [2]
    assert metrics_11["router_step"] == 2
    assert metrics_22["router_step"] == 1
    assert metrics_11["server_policy_action_sha256"] != metrics_22["server_policy_action_sha256"]


def test_reset_requires_a_new_episode_seed(monkeypatch):
    monkeypatch.setattr(server_module, "decode_images", lambda images: images)
    service = server_module.RobotwinPolicyService(_FakeModel())
    service.infer_session(11, _request("reset"))
    with pytest.raises(RuntimeError, match="seed must be set"):
        service.infer_session(11, _infer_request())


def test_native_seed_then_reset_order_preserves_episode_rng(monkeypatch):
    monkeypatch.setattr(server_module, "decode_images", lambda images: images)
    service = server_module.RobotwinPolicyService(_FakeModel())
    service.infer_session(11, _request("seed", seed=101))
    service.infer_session(11, _request("reset"))

    actual = service.infer_session(11, _infer_request())["actions"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        expected = torch.rand((2, 3)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_service_applies_seed_even_when_model_seed_hook_is_a_noop(monkeypatch):
    monkeypatch.setattr(server_module, "decode_images", lambda images: images)
    model = _FakeModel()
    model.set_episode_seed = lambda seed: None
    service = server_module.RobotwinPolicyService(model)

    service.infer_session(11, _request("reset"))
    service.infer_session(11, _request("seed", seed=101))
    actual = service.infer_session(11, _infer_request())["actions"]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        expected = torch.rand((2, 3)).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_close_session_discards_mutable_state():
    service = server_module.RobotwinPolicyService(_FakeModel())
    service.infer_session(11, _request("reset"))
    assert 11 in service._sessions
    service.close_session(11)
    assert 11 not in service._sessions
