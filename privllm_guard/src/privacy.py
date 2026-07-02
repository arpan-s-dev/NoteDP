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


class PrivacyBudgetTracker:
    """§IV.D Eqs. 18–19 — running ε and sliding-window spike detector.

    Eq. 18: ε_remaining(t) = ε_initial − Σ_{i=1}^t ε_i
    Eq. 19: ε_window(t) = Σ_{i=max(1,t−w)}^t ε_i   with w = 50
    Spike: ε_window > 0.15 · ε_total  (§IV.D)
    """

    def __init__(self, total_budget: float, window_size: int = 50) -> None:
        self.total_budget = total_budget
        self.window_size = window_size
        self.consumed = 0.0
        self.step_costs: list[float] = []

    def consume(self, cost: float) -> None:
        self.consumed += cost
        self.step_costs.append(cost)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.consumed)

    @property
    def window_cost(self) -> float:
        recent = self.step_costs[-self.window_size :]
        return float(sum(recent))

    def is_budget_exceeded(self) -> bool:
        return self.remaining <= 0.0

    def is_window_spike(self, threshold_fraction: float = 0.15) -> bool:
        return self.window_cost > threshold_fraction * self.total_budget

    def reset(self) -> None:
        self.consumed = 0.0
        self.step_costs = []


@dataclass
class NoiseParameters:
    """Cached per-sequence noise scales from Adaptive Noise Calibration (§III)."""

    sigma_emb: float
    sigma_att: float
    profile: str  # "high" | "medium" | "low"


class AdaptiveNoiseCalibrator:
    """§III Algorithm 1 prose — ANC runs once per sequence, not per token.

    A single sensitivity profile (high/medium/low) selects (σ_emb, σ_att)
    which are cached for the full generation. This is O(κ) with κ ≈ 512,
    not O(n·κ) per token.

    [UNSPECIFIED] The paper never states the numeric mapping from profile
    to σ. Using multipliers on Appendix A base σ_emb=0.5, σ_att=0.3.
    Alternatives: learned RL policy (Eq. 10 prose, algorithm unspecified).
    """

    def __init__(
        self,
        sigma_emb: float,
        sigma_att: float,
        multipliers: dict[str, float] | None = None,
    ) -> None:
        self.base_sigma_emb = sigma_emb
        self.base_sigma_att = sigma_att
        self.multipliers = multipliers or {"high": 1.5, "medium": 1.0, "low": 0.5}

    def from_profile(self, profile: str) -> NoiseParameters:
        scale = self.multipliers[profile]
        return NoiseParameters(
            sigma_emb=self.base_sigma_emb * scale,
            sigma_att=self.base_sigma_att * scale,
            profile=profile,
        )

    def from_probs(self, profile_probs: torch.Tensor) -> NoiseParameters:
        """profile_probs: (3,) or (batch, 3) softmax over [low, medium, high]."""
        if profile_probs.dim() == 2:
            profile_probs = profile_probs.mean(dim=0)
        idx = int(torch.argmax(profile_probs).item())
        names = ("low", "medium", "high")
        return self.from_profile(names[idx])

    def emergency_recalibrate(self, current: NoiseParameters) -> NoiseParameters:
        """§III — if the monitor flags a violation, increase noise (stricter).

        [UNSPECIFIED] Recalibration rule not given. Using: bump one profile
        level toward 'high' (more noise).
        """
        order = ("low", "medium", "high")
        idx = min(order.index(current.profile) + 1, len(order) - 1)
        return self.from_profile(order[idx])


def hierarchical_layer_sigmas(
    base_sigma_att: float,
    n_layers: int,
) -> list[float]:
    """§III — 'varied layers possess varying privacy budgets'.

    [PARTIALLY_SPECIFIED] No per-layer schedule is given. Using linearly
    *decreasing* attention noise with depth so later layers (more abstract
    clinical concepts) keep more utility, matching the hierarchical story
    in §V Performance drivers. Alternatives: uniform σ (official code).
    """
    if n_layers <= 1:
        return [base_sigma_att]
    scales = torch.linspace(1.15, 0.85, n_layers)
    return [base_sigma_att * float(s) for s in scales]


def exponential_mechanism_sample(
    utilities: torch.Tensor,
    epsilon_t: float,
    delta_u: float,
) -> torch.Tensor:
    """§IV.B, Eq. 7 — private token sampling.

    P(w_t | w_<t) ∝ exp( ε_t · u(w_t, w_<t) / (2 Δu) )

    Args:
        utilities: (..., vocab) utility scores. We use LM logits as u(·).
        epsilon_t: per-token privacy budget ε_t
        delta_u: sensitivity of u. Appendix A: 'empirically tuned per task'.

    Returns:
        sampled token ids, shape (...,)
    """
    # [UNSPECIFIED] Paper does not say u is logits. Using logits as utility.
    # Alternatives: log-softmax; clipped logits.
    scale = epsilon_t / (2.0 * max(delta_u, 1e-8))
    log_probs = utilities * scale  # (..., vocab)
    probs = F.softmax(log_probs, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return idx.reshape(utilities.shape[:-1])


class RealTimePrivacyMonitor:
    """§IV.D — per-token O(w) accumulator plus risk score R(y) (Eq. 14).

    Eq. 14: R(y) = Σ_i w_i · P(leak_i | y_{≤i})
    The monitor does not run the heavy ANC; it only accumulates budget and
    risk. Emergency recalibration is triggered on budget exhaustion or a
    window spike (§III: <2.3% of sequences in the paper).
    """

    def __init__(
        self,
        tracker: PrivacyBudgetTracker,
        spike_fraction: float = 0.15,
        risk_threshold: float = 0.5,
    ) -> None:
        # [UNSPECIFIED] R(y) halt threshold is 'a predefined threshold'.
        # Using 0.5 on the mean token leak probability scale.
        self.tracker = tracker
        self.spike_fraction = spike_fraction
        self.risk_threshold = risk_threshold

    def risk_score(
        self,
        leak_probs: torch.Tensor,
        token_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Eq. 14.

        Args:
            leak_probs: P(leak_i | y_{≤i}) — (batch, seq_len)
            token_weights: w_i — (batch, seq_len) or (seq_len,)

        Returns:
            R(y) per sequence — (batch,)
        """
        if token_weights is None:
            token_weights = torch.ones_like(leak_probs)
        if token_weights.dim() == 1:
            token_weights = token_weights.unsqueeze(0).expand_as(leak_probs)
        return (token_weights * leak_probs).sum(dim=-1)  # (batch,)

    def should_recalibrate(self, sequence_risk: torch.Tensor | None = None) -> bool:
        if self.tracker.is_budget_exceeded():
            return True
        if self.tracker.is_window_spike(self.spike_fraction):
            return True
        if sequence_risk is not None and bool((sequence_risk > self.risk_threshold).any()):
            return True
        return False
