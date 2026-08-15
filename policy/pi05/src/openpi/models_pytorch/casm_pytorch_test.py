from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch import casm_pytorch
from openpi.models_pytorch import pi0_pytorch


def test_reduce_action_loss_applies_mask_and_ignores_fully_masked_step():
    squared_error = torch.tensor([[[1.0, 9.0], [4.0, 16.0]]])
    action_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    loss = casm_pytorch.reduce_action_loss(squared_error, action_mask)

    assert torch.equal(loss, torch.tensor([[1.0, 0.0]]))


def test_reduce_action_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="action mask shape mismatch"):
        casm_pytorch.reduce_action_loss(torch.ones(1, 2, 3), torch.ones(1, 2))


def test_visual_gate_starts_at_half_probability_and_detaches_inputs():
    gate = casm_pytorch.VisualProprioceptionGate(4, 3, 5)
    visual = torch.randn(2, 4, requires_grad=True)
    state = torch.randn(2, 3, requires_grad=True)

    logits = gate(visual, state)
    logits.sum().backward()

    assert torch.equal(logits, torch.zeros(2))
    assert visual.grad is None
    assert state.grad is None
    assert gate.output.weight.grad is not None


def test_visual_gate_casts_inputs_to_parameter_dtype():
    gate = casm_pytorch.VisualProprioceptionGate(4, 3, 5).to(torch.float64)
    visual = torch.randn(2, 4, dtype=torch.float32)
    state = torch.randn(2, 3, dtype=torch.float32)

    logits = gate(visual, state)

    assert logits.dtype == torch.float64
    assert torch.equal(logits, torch.zeros(2, dtype=torch.float64))


def test_visual_phase_gate_loss_matches_weighted_bce_and_metrics():
    action_loss = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    logits = torch.zeros(2)
    phase_id = torch.tensor([[1], [0]])

    result = casm_pytorch.visual_phase_gate_loss(
        action_loss,
        logits,
        phase_id,
        gate_loss_weight=0.2,
        gate_positive_weight=3.0,
    )

    log_two = torch.log(torch.tensor(2.0))
    expected_gate_loss = torch.tensor([3.0, 1.0]) * log_two
    assert torch.allclose(result.total, action_loss + 0.2 * expected_gate_loss[:, None])
    assert torch.allclose(result.metrics["action_loss"], torch.tensor(2.5))
    assert torch.allclose(result.metrics["gate_loss"], expected_gate_loss.mean())
    assert torch.allclose(result.metrics["gate_accuracy"], torch.tensor(0.5))
    assert torch.allclose(result.metrics["gate_async_probability"], torch.tensor(0.5))
    assert torch.allclose(result.metrics["gate_predicted_async_rate"], torch.tensor(1.0))
    assert torch.allclose(result.metrics["gate_target_async_rate"], torch.tensor(0.5))


def test_visual_phase_gate_loss_backpropagates_to_gate_logits():
    logits = torch.zeros(2, requires_grad=True)
    result = casm_pytorch.visual_phase_gate_loss(
        torch.zeros(2, 3),
        logits,
        torch.tensor([[1], [0]]),
        gate_loss_weight=0.2,
        gate_positive_weight=1.0,
    )

    result.total.mean().backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 2


@pytest.mark.parametrize("positive_weight", [0.0, -1.0])
def test_visual_phase_gate_loss_rejects_invalid_positive_weight(positive_weight):
    with pytest.raises(ValueError, match="positive weight must be positive"):
        casm_pytorch.visual_phase_gate_loss(
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.zeros(1, 1, dtype=torch.int64),
            gate_loss_weight=0.2,
            gate_positive_weight=positive_weight,
        )


def test_subtask_classifier_detaches_inputs_but_updates_head():
    classifier = casm_pytorch.VisualProprioceptionClassifier(
        visual_dim=4,
        state_dim=3,
        hidden_dim=5,
        classes=3,
        stop_gradient=True,
    )
    visual = torch.randn(2, 4, requires_grad=True)
    state = torch.randn(2, 3, requires_grad=True)

    result = casm_pytorch.subtask_classification_loss(
        classifier(visual, state),
        torch.tensor([[0], [2]]),
    )
    result.per_sample.mean().backward()

    assert visual.grad is None
    assert state.grad is None
    assert classifier.output.weight.grad is not None
    assert torch.isfinite(result.per_sample).all()


def test_subtask_classification_loss_reports_cross_entropy_and_accuracy():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

    result = casm_pytorch.subtask_classification_loss(logits, torch.tensor([[0], [0]]))

    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0]), reduction="none")
    assert torch.allclose(result.per_sample, expected)
    assert result.metrics["subtask_loss"] == pytest.approx(expected.mean())
    assert result.metrics["subtask_accuracy"] == pytest.approx(0.5)


def test_subtask_classification_loss_normalizes_by_selected_class_weight():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    target = torch.tensor([0, 1, 1])

    result = casm_pytorch.subtask_classification_loss(logits, target, class_weights=(0.5, 2.0))

    negative_log_likelihood = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    selected_weights = torch.tensor([0.5, 2.0, 2.0])
    expected_per_sample = negative_log_likelihood * selected_weights
    assert torch.allclose(result.per_sample, expected_per_sample)
    assert result.metrics["subtask_loss"] == pytest.approx(expected_per_sample.sum() / selected_weights.sum())
    assert result.metrics["subtask_unweighted_loss"] == pytest.approx(negative_log_likelihood.mean())


def test_subtask_classification_loss_rejects_out_of_range_target():
    with pytest.raises(ValueError, match="outside"):
        casm_pytorch.subtask_classification_loss(torch.zeros(1, 2), torch.tensor([[2]]))


def test_pi0_subtask_only_forward_does_not_require_actions(monkeypatch):
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.aux_subtask_classes = 2
    model.aux_subtask_state_dim = 3
    model.aux_subtask_class_weights = None
    model.subtask_head = casm_pytorch.VisualProprioceptionClassifier(
        visual_dim=4,
        state_dim=3,
        hidden_dim=5,
        classes=2,
        stop_gradient=True,
    )
    visual = torch.randn(2, 4)
    state = torch.randn(2, 5)
    observation = SimpleNamespace(semantic_subtask_id=torch.tensor([[0], [1]]))

    monkeypatch.setattr(
        model,
        "_preprocess_observation",
        lambda unused, train: ([None], [None], None, None, state),
    )
    monkeypatch.setattr(
        model,
        "embed_prefix",
        lambda images, img_masks, lang_tokens, lang_masks: (None, None, None, visual),
    )

    metrics = model(observation, subtask_only=True)

    assert torch.isfinite(metrics["subtask_loss"])
    assert metrics["subtask_accuracy"] >= 0
