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
    cfg: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | None = None,
    use_teacher: bool = True,
) -> dict[str, Any]:
    """Full training loop (full mode)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["training"]["seed"]))
    model = PrivLLMGuard(ModelConfig.from_dict(cfg)).to(device)
    teacher = None
    if use_teacher:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    optimizer = build_optimizer(model, cfg)
    # [FROM_OFFICIAL_CODE] CosineAnnealingLR over epochs. Paper unspecified.
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["epochs"]))
    weights = LossWeights.from_dict(cfg)
    clipper = AdaptiveGradientClipping(
        initial_clip=float(cfg["privacy"]["clip_C"]),
        momentum=float(cfg["privacy"]["clip_momentum"]),
        quantile=float(cfg["privacy"]["clip_quantile_p"]),
    )
    accountant = RDPAccountant(cfg["privacy"]["rdp_orders"])
    epochs = int(cfg["training"]["epochs"])
    n_steps = max(1, epochs * len(train_loader))
    # Spend the decoder training budget across optimizer steps (Eq. 9 ε_dec).
    epsilon_step = float(cfg["privacy"]["epsilon_dec"]) / n_steps
    delta = float(cfg["privacy"]["delta"])
    patience = int(cfg["training"]["early_stop_patience"])
    best_val = float("inf")
    stale = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "epsilon": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, sigma = dp_sgd_microbatch(
                model, batch, teacher, weights, clipper, epsilon_step, delta, device
            )
            optimizer.step()
            accountant.accumulate(sigma=sigma, sensitivity=clipper.clip_bound)
            running += float(loss.item())
            n += 1
        scheduler.step()
        train_loss = running / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, weights, device)
        eps = accountant.get_epsilon(delta)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epsilon"].append(eps)
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return {"model": model, "teacher": teacher, "history": history, "accountant": accountant}


@torch.no_grad()
def evaluate_loss(
    model: PrivLLMGuard,
    loader: DataLoader,
    weights: LossWeights,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        entity_labels = batch["entity_labels"].to(device)
        dec_in = batch["decoder_input_ids"].to(device)
        out = model(
            input_ids,
            decoder_input_ids=dec_in,
            labels=labels,
            apply_noise=False,
        )
        parts = combined_loss(
            out["logits"],
            labels,
            out["leak_probs"],
            out["entity_logits"],
            entity_labels,
            weights,
        )
        total += float(parts["loss"].item())
        n += 1
    return total / max(n, 1)


def main(config_path: str = "configs/walkthrough.yaml") -> None:
    from src.data import stratified_splits

    cfg = load_config(str(Path(config_path)))
    train_ds, val_ds, _ = stratified_splits(
        num_samples=32,
        max_len=int(cfg["model"]["max_seq_len"]),
        vocab_size=int(cfg["model"]["vocab_size"]),
        seed=int(cfg["training"]["seed"]),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]))
    result = train(cfg, train_loader, val_loader, use_teacher=False)
    print(result["history"])


if __name__ == "__main__":
    main()
