import torch
import torch.nn.functional as F

from rubric_dpo.losses import baseline_losses, mmpo_loss_from_p0, scaled_dpo_hs2_native

DTYPE = torch.float64


def tensors():
    return (
        torch.tensor([-2.1, -3.2, -1.7], dtype=DTYPE, requires_grad=True),
        torch.tensor([-2.8, -2.9, -2.4], dtype=DTYPE, requires_grad=True),
        torch.tensor([-2.0, -3.4, -1.9], dtype=DTYPE),
        torch.tensor([-2.7, -3.0, -2.1], dtype=DTYPE),
    )


def test_dpo_matches_logsigmoid_formula_at_fp64():
    pc, pr, rc, rr = tensors()
    beta = 0.01
    actual = baseline_losses("dpo", pc, pr, rc, rr, beta=beta).losses
    h = (pc - pr) - (rc - rr)
    expected = -F.logsigmoid(beta * h)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def test_mmpo_matches_soft_label_bce_and_p0_one_is_dpo():
    z = torch.tensor([-1.0, 0.2, 3.0], dtype=DTYPE)
    p0 = torch.tensor([0.6, 0.8, 0.95], dtype=DTYPE)
    expected = -(p0 * F.logsigmoid(z) + (1 - p0) * F.logsigmoid(-z))
    torch.testing.assert_close(mmpo_loss_from_p0(z, p0), expected, atol=1e-10, rtol=0)
    torch.testing.assert_close(mmpo_loss_from_p0(z, torch.ones_like(z)), F.softplus(-z), atol=0, rtol=0)


def test_odpo_matches_author_loggap_and_alpha_zero_is_dpo_value_and_gradient():
    pc, pr, rc, rr = tensors()
    gap = torch.tensor([0.1, 0.5, 1.0], dtype=DTYPE)
    beta, alpha = 0.01, 0.5
    actual = baseline_losses("odpo_loggap", pc, pr, rc, rr, beta=beta, margin_normalized=gap, alpha=alpha).losses
    h = (pc - pr) - (rc - rr)
    expected = -F.logsigmoid(beta * h - alpha * torch.log(gap))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    dpo = baseline_losses("dpo", pc, pr, rc, rr, beta=beta).losses.sum()
    dpo_grad = torch.autograd.grad(dpo, (pc, pr), retain_graph=True)
    odpo0 = baseline_losses("odpo_loggap", pc, pr, rc, rr, beta=beta, margin_normalized=gap, alpha=0).losses.sum()
    odpo_grad = torch.autograd.grad(odpo0, (pc, pr))
    torch.testing.assert_close(dpo, odpo0, atol=1e-12, rtol=0)
    for left, right in zip(dpo_grad, odpo_grad):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)


def test_scaled_unit_weight_is_dpo_and_native_ratios_apply_to_loss_and_gradient():
    pc, pr, rc, rr = tensors()
    ones = torch.ones(3, dtype=DTYPE)
    dpo = baseline_losses("dpo", pc, pr, rc, rr, beta=0.01).losses
    scaled = baseline_losses("scaled_dpo_gap_transfer", pc, pr, rc, rr, beta=0.01, sample_weight=ones).losses
    torch.testing.assert_close(dpo, scaled, atol=0, rtol=0)
    z = torch.tensor([0.4, 0.4, 0.4], dtype=DTYPE, requires_grad=True)
    strength = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    losses = scaled_dpo_hs2_native(z, strength)
    grads = torch.autograd.grad(losses.sum(), z)[0]
    torch.testing.assert_close(losses / losses[0], strength, atol=1e-12, rtol=0)
    torch.testing.assert_close(grads / grads[0], strength, atol=1e-12, rtol=0)


def test_token_sum_is_not_token_mean():
    token_logps = torch.tensor([[-1.0, -2.0, 0.0], [-1.0, -2.0, -3.0]], dtype=DTYPE)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    sums = (token_logps * mask).sum(-1)
    means = sums / mask.sum(-1)
    assert not torch.equal(sums, means)


def test_equal_local_batches_reduce_to_global_arithmetic_mean():
    losses = torch.tensor([0.1, 0.4, 1.2, 0.8, 0.3, 2.0, 0.9, 0.7], dtype=DTYPE)
    rank_means = torch.stack([chunk.mean() for chunk in losses.chunk(4)])
    torch.testing.assert_close(rank_means.mean(), losses.mean(), atol=1e-12, rtol=0)
