import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.online_subtask import next_token_cross_entropy


def test_next_token_cross_entropy_uses_only_target_positions():
    tokens = torch.tensor([[4, 2, 1, 3]])
    loss_mask = torch.tensor([[False, False, True, True]])
    logits = torch.zeros(1, 3, 5)
    logits[0, 1, 1] = 2.0
    logits[0, 2, 3] = 3.0

    actual = next_token_cross_entropy(logits, tokens, loss_mask)
    expected = F.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2]]),
        torch.tensor([1, 3]),
    )

    torch.testing.assert_close(actual, expected)


def test_next_token_cross_entropy_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        next_token_cross_entropy(
            torch.zeros(1, 2, 5),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 4, dtype=torch.bool),
        )


def test_next_token_cross_entropy_rejects_empty_target():
    with pytest.raises(ValueError, match="target mask is empty"):
        next_token_cross_entropy(
            torch.zeros(1, 2, 5),
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.bool),
        )


def test_feature_off_forward_delegates_to_standard_action_path_exactly():
    expected = torch.tensor([[1.0, 2.0]])
    captured = {}

    class FeatureOff:
        online_subtask_prediction = False

        def _action_loss(
            self,
            observation,
            actions,
            noise=None,
            time=None,
            *,
            return_aux=False,
            preprocessed=None,
        ):
            captured.update(
                observation=observation,
                actions=actions,
                noise=noise,
                time=time,
                return_aux=return_aux,
            )
            return expected

    observation = object()
    actions = torch.ones(1, 2)
    noise = torch.zeros(1, 2)
    time = torch.tensor([0.5])
    actual = pi0_pytorch.PI0Pytorch.forward(
        FeatureOff(),
        observation,
        actions,
        noise=noise,
        time=time,
        return_aux=True,
    )

    assert actual is expected
    assert captured == {
        "observation": observation,
        "actions": actions,
        "noise": noise,
        "time": time,
        "return_aux": True,
    }


def test_feature_on_teacher_forces_action_prompt_and_combines_losses():
    class Observation:
        action_mask = None
        tokenized_prompt = torch.tensor([[1, 2]])
        tokenized_prompt_mask = torch.tensor([[True, True]])
        tokenized_action_prompt = torch.tensor([[3, 4]])
        tokenized_action_prompt_mask = torch.tensor([[True, True]])

        def replace(self, **updates):
            result = Observation()
            for name, value in updates.items():
                setattr(result, name, value)
            return result

    class FeatureOn:
        online_subtask_prediction = True
        casm_mode = "none"
        lambda_subtask = 0.25
        _reduce_action_objective = (
            pi0_pytorch.PI0Pytorch._reduce_action_objective  # noqa: SLF001
        )
        seen_prompt = None
        preprocess_calls = 0

        def _preprocess_observation(self, observation, *, train):
            self.preprocess_calls += 1
            assert train
            return ((torch.tensor(1.0),), (torch.ones((), dtype=torch.bool),), None, None, None)

        def _action_loss(
            self,
            observation,
            actions,
            noise=None,
            time=None,
            *,
            return_aux=False,
            preprocessed=None,
        ):
            del actions, noise, time, return_aux, preprocessed
            self.seen_prompt = observation.tokenized_prompt
            return torch.tensor([[2.0]], requires_grad=True)

        def _subtask_ce(self, observation, images, image_masks):
            del observation, images, image_masks
            return torch.tensor(4.0, requires_grad=True)

    model = FeatureOn()
    observation = Observation()
    output = pi0_pytorch.PI0Pytorch.forward(
        model,
        observation,
        torch.zeros(1, 1),
    )

    assert output.total.detach().item() == pytest.approx(3.0)
    assert output.action.detach().item() == pytest.approx(2.0)
    assert output.subtask_ce.detach().item() == pytest.approx(4.0)
    assert model.seen_prompt is observation.tokenized_action_prompt
    assert model.preprocess_calls == 1


def test_next_token_cross_entropy_is_fullgraph_traceable():
    compiled = torch.compile(
        next_token_cross_entropy,
        backend="eager",
        fullgraph=True,
    )
    tokens = torch.tensor([[4, 2, 1, 3]])
    loss_mask = torch.tensor([[False, False, True, True]])
    logits = torch.zeros(1, 3, 5)

    actual = compiled(logits, tokens, loss_mask)

    assert torch.isfinite(actual)


def test_action_objective_is_fullgraph_traceable_with_mask():
    class Reducer:
        _reduce_action_objective = pi0_pytorch.PI0Pytorch._reduce_action_objective  # noqa: SLF001

    def reduce_action_objective(action_loss, action_mask):
        return Reducer()._reduce_action_objective(action_loss, action_mask)  # noqa: SLF001

    compiled = torch.compile(
        reduce_action_objective,
        backend="eager",
        fullgraph=True,
    )
    action_loss = torch.tensor([[1.0, 3.0]])
    action_mask = torch.tensor([[[True, True], [True, False]]])

    actual = compiled(action_loss, action_mask)

    assert actual.item() == pytest.approx(5.0 / 3.0)
