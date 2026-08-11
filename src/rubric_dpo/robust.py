from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class RobustOutput:
    loss: torch.Tensor
    adversarial_distribution: torch.Tensor
    diagnostics: dict[str, float]


class RobustDistanceBackend(Protocol):
    """Reserved interface; baseline phase intentionally provides no solver."""

    name: str

    def solve(
        self,
        nominal_distribution: torch.Tensor,
        branch_losses: torch.Tensor,
        rubric_s: torch.Tensor,
        rho: float,
    ) -> RobustOutput: ...


RESERVED_BACKENDS = ("kl", "wasserstein_w1")
