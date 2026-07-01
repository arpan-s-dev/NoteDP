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

