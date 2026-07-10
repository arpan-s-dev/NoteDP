"""
PrivLLM-Guard — loss functions.

Paper: https://doi.org/10.1038/s41598-026-45883-6

  Eq. 8  L_distill = Σ KL(P_teacher(y_i|x) || P_student(y_i|x))
  Eq. 12 L = L_LM + λ1 L_privacy + λ2 L_utility + λ3 L_medical
  Eq. 13 L_coherence = −Σ log P(s_{i+1} | s_{≤i})

Eq. 10 (α L_utility + β L_privacy + γ L_coherence with RL-learned weights)
is not implemented as RL: Appendix A gives fixed λ, and the RL algorithm
is unspecified. We use Eq. 12 with Appendix A λ and Eq. 13 as L_utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_privacy: float = 1.0  # Appendix A λ1
    lambda_utility: float = 1.0  # Appendix A λ2
    lambda_medical: float = 0.5  # Appendix A λ3
    distill_temperature: float = 2.0  # Appendix A

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "LossWeights":
        t = cfg["training"]
        return cls(
            lambda_privacy=float(t["lambda_privacy"]),
            lambda_utility=float(t["lambda_utility"]),
            lambda_medical=float(t["lambda_medical"]),
            distill_temperature=float(t["distill_temperature"]),
        )


def language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """§IV.C — L_LM, standard next-token cross-entropy.

    Args:
        logits: (batch, seq, vocab)
        labels: (batch, seq)
    """
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def privacy_risk_loss(leak_probs: torch.Tensor) -> torch.Tensor:
    """§IV.C L_privacy — mean Eq. 14 leak probability (without w_i).

    [PARTIALLY_SPECIFIED] Paper names L_privacy but does not write a formula.
    Using mean token leak probability from the risk head.
    """
    return leak_probs.mean()


def coherence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """§IV.C, Eq. 13 — L_coherence = −Σ log P(s_{i+1} | s_{≤i}).

    Sentence boundaries are not annotated in the skeleton dataset, so we
    implement the same conditional log-prob on the token chain (a strict
    reading of 'sentence i' would require a sentence splitter).

    Official pllm.py used 1 − cosine similarity of adjacent hidden states,
    which is not Eq. 13. We follow the equation.
    """
    if logits.size(1) < 2:
        return logits.new_zeros(())
    # Predict token t+1 from logits at t.
    pred = logits[:, :-1, :]  # (batch, seq-1, vocab)
    tgt = labels[:, 1:]  # (batch, seq-1)
    return F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        tgt.reshape(-1),
        ignore_index=-100,
    )


def medical_entity_loss(
    entity_logits: torch.Tensor,
    entity_labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """§IV.C L_medical — entity recognition CE on the encoder states."""
    n_types = entity_logits.size(-1)
    return F.cross_entropy(
        entity_logits.reshape(-1, n_types),
        entity_labels.reshape(-1),
        ignore_index=ignore_index,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """§IV.B, Eq. 8 — KL(P_teacher || P_student) with temperature T.

    PyTorch kl_div expects log-input for the student. We use batchmean.
    """
    t = temperature
    log_student = F.log_softmax(student_logits / t, dim=-1)
    teacher_probs = F.softmax(teacher_logits / t, dim=-1)
    # kl_div sums over vocab then averages batch*seq
    kl = F.kl_div(log_student, teacher_probs, reduction="batchmean")
    return kl * (t * t)


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    leak_probs: torch.Tensor,
    entity_logits: torch.Tensor | None,
    entity_labels: torch.Tensor | None,
    weights: LossWeights,
    teacher_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """§IV.C, Eq. 12 — L = L_LM + λ1 L_privacy + λ2 L_utility + λ3 L_medical.

    L_utility is instantiated as L_coherence (Eq. 13) because Eq. 12 does
    not define L_utility separately. Distillation (Eq. 8) is added when a
    teacher is provided (utility recovery, §IV.B).
    """
    loss_lm = language_modeling_loss(logits, labels)
    loss_priv = privacy_risk_loss(leak_probs)
    loss_util = coherence_loss(logits, labels)
    loss_med = logits.new_zeros(())
    if entity_logits is not None and entity_labels is not None:
        loss_med = medical_entity_loss(entity_logits, entity_labels)
    loss_distill = logits.new_zeros(())
    if teacher_logits is not None:
        loss_distill = distillation_loss(
            logits, teacher_logits, temperature=weights.distill_temperature
        )
    total = (
        loss_lm
        + weights.lambda_privacy * loss_priv
        + weights.lambda_utility * loss_util
        + weights.lambda_medical * loss_med
        + loss_distill
    )
    return {
        "loss": total,
        "loss_lm": loss_lm.detach(),
        "loss_privacy": loss_priv.detach(),
        "loss_utility": loss_util.detach(),
        "loss_medical": loss_med.detach(),
        "loss_distill": loss_distill.detach()
        if isinstance(loss_distill, torch.Tensor)
        else loss_distill,
    }
