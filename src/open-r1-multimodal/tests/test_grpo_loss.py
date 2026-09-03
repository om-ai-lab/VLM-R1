import pytest
import torch

from open_r1.grpo_loss import aggregate_grpo_loss


def test_grpo_and_dr_grpo_reductions_have_different_length_scaling():
    per_token_loss = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    completion_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    # The historical GRPO reduction averages the two per-completion means.
    assert aggregate_grpo_loss(per_token_loss, completion_mask).item() == pytest.approx(1.0)
    # Dr. GRPO keeps the configured maximum length in the denominator.
    assert aggregate_grpo_loss(
        per_token_loss,
        completion_mask,
        loss_type="dr_grpo",
        max_completion_length=4,
    ).item() == pytest.approx(0.75)


def test_bnpo_normalizes_by_active_tokens_and_ignores_padding():
    per_token_loss = torch.tensor([[2.0, -1.0, 100.0], [3.0, 0.0, 50.0]])
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    assert aggregate_grpo_loss(per_token_loss, completion_mask, loss_type="bnpo").item() == pytest.approx(1.0)


@pytest.mark.parametrize("loss_type", ["invalid", "DR_GRPO", ""])
def test_unknown_loss_type_is_rejected(loss_type):
    with pytest.raises(ValueError, match="Unknown loss type"):
        aggregate_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), loss_type=loss_type)


def test_dr_grpo_requires_a_positive_maximum_length():
    with pytest.raises(ValueError, match="max_completion_length"):
        aggregate_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), loss_type="dr_grpo")
    with pytest.raises(ValueError, match="max_completion_length"):
        aggregate_grpo_loss(torch.ones(1, 2), torch.ones(1, 2), loss_type="dr_grpo", max_completion_length=0)
