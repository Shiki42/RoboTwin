import numpy as np
import pytest

from robotwin_image_transport import ROBOTWIN_POLICY_PROTOCOL
from robotwin_image_transport import decode_images
import robotwin_remote_model


class FakeClient:
    def __init__(self):
        self.requests = []

    def get_server_metadata(self):
        return {"protocol": ROBOTWIN_POLICY_PROTOCOL}

    def infer(self, request):
        self.requests.append(request)
        if request["command"] == "infer":
            return {"actions": np.ones((3, 14), dtype=np.float32)}
        if request["command"] == "metrics":
            return {"metrics": {"chunk_count": 1}}
        return {"ok": True}


def test_remote_model_forwards_episode_protocol(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(robotwin_remote_model, "_client_policy", lambda host, port: client)
    model = robotwin_remote_model.RobotwinRemoteModel("localhost", 8000)
    images = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3
    state = np.zeros(14, dtype=np.float32)

    model.set_episode_seed(100008)
    model.reset_obsrvationwindows()
    model.set_language("put the object in the cabinet")
    model.update_observation_window(images, state)
    actions = model.get_action()
    model.record_chunk(3)
    model.record_action(actions[0], state)
    model.update_observation_window(images, state + 1, action_executed=True)

    assert [request["command"] for request in client.requests] == ["seed", "reset", "infer", "observe"]
    assert model.execution_steps() == 3
    assert model.inference_index == 1
    assert len(model.episode_action_sha256()) == 64
    assert np.array_equal(client.requests[-1]["previous_state"], state)
    decoded = decode_images(client.requests[-1]["images"])
    assert all(
        np.array_equal(image, expected)
        for image, expected in zip(
            decoded,
            images,
            strict=True,
        )
    )
    assert model.rollout_metrics() == {"chunk_count": 1}


def test_remote_model_rejects_observe_without_action(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(robotwin_remote_model, "_client_policy", lambda host, port: client)
    model = robotwin_remote_model.RobotwinRemoteModel("localhost", 8000)
    images = [np.zeros((4, 4, 3), dtype=np.uint8)] * 3

    with pytest.raises(RuntimeError, match="record_action"):
        model.update_observation_window(images, np.zeros(14), action_executed=True)


def test_remote_model_rejects_wrong_server_protocol(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(
        client,
        "get_server_metadata",
        lambda: {"protocol": "robotwin_pi0_v2"},
    )
    monkeypatch.setattr(
        robotwin_remote_model,
        "_client_policy",
        lambda host, port: client,
    )

    with pytest.raises(ValueError, match="protocol mismatch"):
        robotwin_remote_model.RobotwinRemoteModel("localhost", 8000)
