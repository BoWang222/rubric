from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import VARIANTS


@dataclass(frozen=True)
class BaselineLossOutput:
    losses: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    logits: torch.Tensor


def mmpo_loss_from_p0(z: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(p0).all() or not ((p0 >= 0) & (p0 <= 1)).all():
        raise ValueError("MMPO p0 must be finite and in [0,1]")
    return p0 * F.softplus(-z) + (1.0 - p0) * F.softplus(z)


def preference_logits(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return chosen_rewards - rejected_rewards, chosen_rewards.detach(), rejected_rewards.detach()


def baseline_losses(
    variant: str,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float,
    margin_normalized: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    gamma: float = 2.2,
    alpha: float = 0.5,
    epsilon: float = 1e-6,
) -> BaselineLossOutput:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown baseline variant {variant!r}; expected one of {VARIANTS}")
    z, chosen_rewards, rejected_rewards = preference_logits(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )
    hard = F.softplus(-z)
    if variant == "dpo":
        losses = hard
    elif variant == "mmpo":
        if margin_normalized is None:
            raise ValueError("MMPO requires margin_normalized")
        p0 = torch.sigmoid(gamma * margin_normalized).clamp(epsilon, 1.0 - epsilon)
        losses = mmpo_loss_from_p0(z, p0)
    elif variant == "odpo_loggap":
        if margin_normalized is None:
            raise ValueError("ODPO requires a positive margin_normalized gap")
        if not torch.isfinite(margin_normalized).all() or not (margin_normalized > 0).all():
            raise ValueError("ODPO gap must be finite and strictly positive")
        # Author code uses beta*h - alpha*log(gap); the offset is outside beta.
        losses = F.softplus(-(z - alpha * torch.log(margin_normalized)))
    else:
        if sample_weight is None:
            raise ValueError("Scaled DPO requires train-normalized sample_weight")
        if not torch.isfinite(sample_weight).all() or not (sample_weight > 0).all():
            raise ValueError("Scaled DPO weights must be finite and positive")
        losses = sample_weight * hard
    return BaselineLossOutput(losses, chosen_rewards, rejected_rewards, z)


def scaled_dpo_hs2_native(z: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """Paper-formula parity fixture. This is not the UltraFeedback main lane."""
    if not torch.isin(strength, torch.tensor([1, 2, 3], device=strength.device, dtype=strength.dtype)).all():
        raise ValueError("HelpSteer2 native Scaled DPO strength must be in {1,2,3}")
    return strength * F.softplus(-z)
