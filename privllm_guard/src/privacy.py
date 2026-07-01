"""
PrivLLM-Guard — differential privacy mechanisms.

Paper: https://doi.org/10.1038/s41598-026-45883-6
Alghamdi, Sci Rep 16:15781 (2026)

Implements §IV mathematical modeling:
  Eq. 1  (ε,δ)-DP definition (documentation only)
  Eq. 2  global L2 sensitivity
  Eq. 6  Gaussian noise scale σ = C √(2 ln(1.25/δ)) / ε
  Eq. 5  per-example gradient clipping
  Eq. 7  exponential mechanism token sampling
  Eq. 9  hierarchical budget (tracked here; split comes from config)
  Eq. 11 adaptive clipping bound
  Eqs. 16–17 RDP composition and conversion
  Eqs. 18–19 remaining budget and sliding window
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


class GaussianMechanism:
    """§IV.A–B — Gaussian mechanism used for embedding, attention, and DP-SGD noise.

    Eq. 1 — (ε,δ)-DP:
        Pr[M(D) ∈ O] ≤ e^ε Pr[M(D') ∈ O] + δ
    """

    @staticmethod
    def l2_sensitivity(f_d: torch.Tensor, f_d_prime: torch.Tensor) -> torch.Tensor:
        """§IV.A, Eq. 2 — Δ_f = max ||f(D) − f(D')||_2 over adjacent datasets.

        For a single adjacent pair this returns ||f(D)−f(D')||_2.
        """
        return torch.norm(f_d - f_d_prime, p=2)

    @staticmethod
    def calibrate_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
        """§IV.B, Eq. 6 — σ = C √(2 ln(1.25/δ)) / ε with C = sensitivity.

        NOTE: Official pllm.py multiplies the result by 0.01. That factor is
        not in Eq. 6. We implement Eq. 6 as written.
        """
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive to calibrate Gaussian noise")
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    @staticmethod
    def add_noise(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add N(0, σ² I) as in Eq. 3 (embeddings) and Eq. 4 (attention scores).

        Args:
            tensor: any shape
            sigma: standard deviation (not variance)

        Returns:
            tensor + noise, same shape
        """
        if sigma <= 0.0:
            return tensor
        noise = torch.randn_like(tensor) * sigma
        return tensor + noise


class AdaptiveGradientClipping:
    """§IV.B Eq. 5 and §IV.C Eq. 11 — clip then adapt C from gradient-norm quantiles.

    Eq. 5:  g̃_i = g_i · min(1, C / ||g_i||_2)
    Eq. 11: C_{t+1} = α_C C_t + (1 − α_C) · quantile({||g_i||_2}, p)
    Appendix A: C_0 = 1.2, p = 0.5, α_C = 0.9
    """

    def __init__(
        self,
        initial_clip: float = 1.2,
        momentum: float = 0.9,
        quantile: float = 0.5,
        history_size: int = 100,
    ) -> None:
        self.clip_bound = initial_clip
        self.momentum = momentum
        self.quantile = quantile
        # [FROM_OFFICIAL_CODE] official uses the last 100 norms; paper does not cap history.
        self.history_size = history_size
        self.grad_norms: deque[float] = deque(maxlen=history_size)

    def clip_tensor(self, gradients: torch.Tensor) -> torch.Tensor:
        """Clip one tensor (used for a packed per-example gradient)."""
        grad_norm = torch.norm(gradients, p=2)
        self.grad_norms.append(float(grad_norm.item()))
        clip_factor = min(1.0, self.clip_bound / (float(grad_norm.item()) + 1e-8))
        return gradients * clip_factor

    def clip_param_grads_(self, parameters: Sequence[torch.nn.Parameter]) -> float:
        """Clip concatenated parameter gradients in-place to bound C (Abadi-style).

        [PARTIALLY_SPECIFIED] Eq. 5 is per-example. When true per-example grads
        are not available, clipping the aggregated gradient is a strictly weaker
        sensitivity control. Prefer `dp_sgd_microbatch` in train.py.
        """
        total_sq = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            total_sq += float(param.grad.data.norm(2).item() ** 2)
        total_norm = math.sqrt(total_sq)
        self.grad_norms.append(total_norm)
        clip_factor = min(1.0, self.clip_bound / (total_norm + 1e-8))
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_factor)
        return total_norm

    def update_clip_bound(self) -> float:
        """§IV.C, Eq. 11 — EMA of the p-quantile of recent gradient norms."""
        if not self.grad_norms:
            return self.clip_bound
        norms = torch.tensor(list(self.grad_norms), dtype=torch.float32)
        q_val = float(torch.quantile(norms, self.quantile).item())
        self.clip_bound = self.momentum * self.clip_bound + (1.0 - self.momentum) * q_val
        return self.clip_bound


class RDPAccountant:
    """§IV.D Eqs. 16–17 — Rényi DP composition for long-sequence generation / DP-SGD.

    For a Gaussian mechanism, the RDP of order α is
        ε_RDP(α) = α · Δ² / (2 σ²)
    [FROM_OFFICIAL_CODE] this closed form is the standard Gaussian RDP
    (Mironov; also used in pllm.py). The paper states composition (Eq. 16)
    and conversion (Eq. 17) but not the per-step Gaussian RDP formula.

    Eq. 16: ε_RDP^{total}(α) = Σ_i ε_RDP^i(α)
    Eq. 17: ε = min_α [ ε_RDP^{total}(α) + ln(1/δ)/(α−1) ]
            α ∈ {2, 4, 8, 16, 32, 64}
    """

    def __init__(self, orders: Sequence[int] | None = None) -> None:
        self.orders = list(orders) if orders is not None else [2, 4, 8, 16, 32, 64]
        self.rdp_budgets = {alpha: 0.0 for alpha in self.orders}

    def accumulate(self, sigma: float, sensitivity: float = 1.0) -> None:
        sigma = max(float(sigma), 1e-12)
        for alpha in self.orders:
            self.rdp_budgets[alpha] += alpha * (sensitivity ** 2) / (2.0 * sigma ** 2)

    def get_epsilon(self, delta: float) -> float:
        """Eq. 17 conversion. Returns the tightest ε over listed α."""
        best = float("inf")
        for alpha in self.orders:
            eps = self.rdp_budgets[alpha] + math.log(1.0 / delta) / (alpha - 1)
            if eps < best:
                best = eps
        return best

    def reset(self) -> None:
        self.rdp_budgets = {alpha: 0.0 for alpha in self.orders}

