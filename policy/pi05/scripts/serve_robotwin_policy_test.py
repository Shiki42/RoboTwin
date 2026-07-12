import numpy as np

from scripts.serve_robotwin_policy import RobotwinPolicyService


class FakeModel:
    def __init__(self):
        self.observation_window = None
        self.calls = []

    def reset_obsrvationwindows(self):
        self.calls.append(("reset",))
        self.observation_window = None

    def rollout_metrics(self):
        return {"chunk_count": 1}

    def record_action(self, action, state):
        self.calls.append(("record_action", action, state))

    def update_observation_window(self, images, state, *, action_executed):
        self.calls.append(("update", action_executed))
        self.observation_window = {"images": images, "state": state}

    def set_language(self, instruction):
        self.calls.append(("language", instruction))

    def get_action(self):
        return np.ones((5, 14), dtype=np.float32)

    def execution_steps(self):
        return 3

    def record_chunk(self, length):
        self.calls.append(("chunk", length))


def test_service_dispatches_infer_observe_metrics_and_reset():
    model = FakeModel()
    service = RobotwinPolicyService(model)
    observation = {"images": [np.zeros((2, 2, 3))] * 3, "state": np.zeros(14)}

    response = service.infer({"command": "infer", "instruction": "task", **observation})
    assert response["actions"].shape == (3, 14)
    assert ("language", "task") in model.calls
    assert ("chunk", 3) in model.calls

    service.infer({"command": "observe", "action": np.ones(14), "previous_state": np.zeros(14), **observation})
    assert ("update", True) in model.calls
    assert service.infer({"command": "metrics"}) == {"metrics": {"chunk_count": 1}}
    assert service.infer({"command": "reset"}) == {"ok": True}
