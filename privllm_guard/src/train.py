"""
PrivLLM-Guard — DP-SGD training loop.

Paper: https://doi.org/10.1038/s41598-026-45883-6
  Eq. 5  per-example clip
  Eq. 6  Gaussian σ
  Eq. 11 adaptive C
  Eqs. 16–17 RDP accounting on each optimizer step

§V: 8×A100 distributed training is out of scope. This loop is single-device.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.loss import LossWeights, combined_loss
from src.model import ModelConfig, PrivLLMGuard
from src.privacy import AdaptiveGradientClipping, GaussianMechanism, RDPAccountant
from src.utils import load_config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: PrivLLMGuard, cfg: dict[str, Any]) -> AdamW:
    """Appendix A — AdamW, lr = 3e-5.

    [FROM_OFFICIAL_CODE] weight_decay=0.01. Paper does not state weight decay.
    """
    t = cfg["training"]
    return AdamW(
        model.parameters(),
        lr=float(t["lr"]),
        betas=tuple(t["betas"]),
        eps=float(t["eps"]),
        weight_decay=float(t["weight_decay"]),
    )


def add_gaussian_to_grads(
    model: PrivLLMGuard,
    clipper: AdaptiveGradientClipping,
    epsilon_step: float,
    delta: float,
) -> float:
    """Eq. 6 — after clipping, add N(0, σ²) to each gradient tensor.

    σ = C √(2 ln(1.25/δ)) / ε_step
    We do NOT apply the unofficial 0.01 multiplier from pllm.py.
    """
    sigma = GaussianMechanism.calibrate_sigma(clipper.clip_bound, epsilon_step, delta)
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.data = GaussianMechanism.add_noise(param.grad.data, sigma)
    return sigma


def dp_sgd_microbatch(
    model: PrivLLMGuard,
    batch: dict[str, torch.Tensor],
    teacher: PrivLLMGuard | None,
    weights: LossWeights,
    clipper: AdaptiveGradientClipping,
    epsilon_step: float,
    delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Eq. 5 per-example clip, then mean, then Eq. 6 noise.

    Microbatching is the standard way to realize per-example clipping when
    Opacus is not a paper dependency. Slow but matches the equation.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    entity_labels = batch["entity_labels"].to(device)
    dec_in = batch["decoder_input_ids"].to(device)
    batch_size = input_ids.size(0)

    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
        param.accumulated_grad = None  # type: ignore[attr-defined]

    teacher_logits = None
    if teacher is not None:
        with torch.no_grad():
            teacher_logits = teacher(
                input_ids,
                decoder_input_ids=dec_in,
                apply_noise=False,
            )["logits"]

    losses = []
    for i in range(batch_size):
        model.zero_grad(set_to_none=True)
        sl = slice(i, i + 1)
        t_logits = None if teacher_logits is None else teacher_logits[sl]
        out = model(
            input_ids[sl],
            decoder_input_ids=dec_in[sl],
            labels=labels[sl],
            apply_noise=True,
        )
        parts = combined_loss(
            out["logits"],
            labels[sl],
            out["leak_probs"],
            out["entity_logits"],
            entity_labels[sl],
            weights,
            teacher_logits=t_logits,
        )
        (parts["loss"] / batch_size).backward()
        losses.append(parts["loss"].detach())
        # Clip this example's grads to C (Eq. 5) and accumulate.
        clipper.clip_param_grads_(list(model.parameters()))
        for param in model.parameters():
            if param.grad is None:
                continue
            if getattr(param, "accumulated_grad", None) is None:
                param.accumulated_grad = param.grad.detach().clone()  # type: ignore[attr-defined]
            else:
                param.accumulated_grad += param.grad.detach()  # type: ignore[attr-defined]

    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        acc = getattr(param, "accumulated_grad", None)
        if acc is None:
            continue
        param.grad = acc
        del param.accumulated_grad

    sigma = add_gaussian_to_grads(model, clipper, epsilon_step, delta)
    clipper.update_clip_bound()
    mean_loss = torch.stack(losses).mean()
    return mean_loss, sigma


def train(
