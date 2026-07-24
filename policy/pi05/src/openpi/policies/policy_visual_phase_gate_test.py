import torch

from openpi.policies import policy


class _VisualPhaseGateModel(torch.nn.Module):
    casm_mode = "visual_phase_gate"

    def sample_actions(self, device, observation):
        raise AssertionError("not used by this constructor test")

    def predict_async_probability(self, observation):
        raise AssertionError("not used by this constructor test")


def test_pytorch_visual_phase_gate_exposes_probability_predictor():
    model = _VisualPhaseGateModel()
    deployed = policy.Policy(model, is_pytorch=True, pytorch_device="cpu")

    assert deployed._predict_async_probability == model.predict_async_probability
