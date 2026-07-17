"""
PrivLLM-Guard — evaluation metrics from §V.

Paper: https://doi.org/10.1038/s41598-026-45883-6

Implements metric *functions* used in Tables 1–3:
  BLEU-4, ROUGE-L, entity accuracy, Privacy Score, latency.
Attack success (MIR/AIR/MES) is only specified as percentages, not as
algorithms — we expose hooks and a simple overlap-based leakage proxy.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import torch

from src.model import PrivLLMGuard
from src.privacy import (
    PrivacyBudgetTracker,
    RealTimePrivacyMonitor,
    exponential_mechanism_sample,
)


def ngram_precision(hyp: Sequence[int], ref: Sequence[int], n: int) -> float:
    """Modified n-gram precision (BLEU component). [UNSPECIFIED] BLEU package."""
    if len(hyp) < n or len(ref) < n:
        return 0.0
    def grams(seq: Sequence[int]) -> list[tuple[int, ...]]:
        return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    hyp_g = grams(hyp)
    ref_g = grams(ref)
    counts: dict[tuple[int, ...], int] = {}
    for g in ref_g:
        counts[g] = counts.get(g, 0) + 1
    match = 0
    for g in hyp_g:
        if counts.get(g, 0) > 0:
            match += 1
            counts[g] -= 1
    return match / len(hyp_g)


def bleu4(hyp: Sequence[int], ref: Sequence[int]) -> float:
    """§V Table 2 — BLEU-4. Geometric mean of precisions 1–4, no brevity penalty if hyp≥ref.

    [UNSPECIFIED] Paper does not name sacrebleu vs NLTK. This is a local
    implementation so the scaffold has no extra dependency.
    """
    import math

    weights = [0.25, 0.25, 0.25, 0.25]
    precs = [ngram_precision(hyp, ref, n) for n in range(1, 5)]
    if min(precs) <= 0.0:
        return 0.0
    score = math.exp(sum(w * math.log(p) for w, p in zip(weights, precs)))
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref) / max(len(hyp), 1))
    return bp * score


def lcs_length(a: Sequence[int], b: Sequence[int]) -> int:
    """Longest common subsequence length for ROUGE-L."""
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def rouge_l(hyp: Sequence[int], ref: Sequence[int]) -> float:
    """§V Table 2 — ROUGE-L F1."""
    if not hyp or not ref:
        return 0.0
    lcs = lcs_length(hyp, ref)
    prec = lcs / len(hyp)
    rec = lcs / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def entity_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """§V — medical entity recognition accuracy."""
    pred = logits.argmax(dim=-1)
    mask = labels != ignore_index
    if int(mask.sum()) == 0:
        return 0.0
    correct = (pred[mask] == labels[mask]).float().sum()
    return float((correct / mask.float().sum()).item())


def privacy_score(
    mir_pct: float,
    air_pct: float,
    mes_pct: float,
    dls: float,
    w1: float = 0.30,
    w2: float = 0.25,
    w3: float = 0.25,
    w4: float = 0.20,
) -> float:
    """§V — Privacy Score on a 0–10 scale.

    Privacy Score = w1·(1−MIR/100)·10 + w2·(1−AIR/100)·10
                  + w3·(1−MES/100)·10 + w4·(1−DLS)·10
    Paper example: MIR=4.2, AIR=2.8, MES=5.7, DLS=0.11 → ≈ 9.3
    """
    return (
        w1 * (1.0 - mir_pct / 100.0) * 10.0
        + w2 * (1.0 - air_pct / 100.0) * 10.0
        + w3 * (1.0 - mes_pct / 100.0) * 10.0
        + w4 * (1.0 - dls) * 10.0
    )


def data_leakage_score(leak_probs: torch.Tensor) -> float:
    """DLS ∈ [0, 1] — §V says 'token-level information leakage probability via Eq. (8)'.

    NOTE: Eq. 8 in the body is distillation KL, not leakage. We treat this as
    a mis-citation and use Eq. 14 leak probabilities (mean).
    """
    return float(leak_probs.mean().item())


def membership_inference_gap(
    member_losses: Iterable[float],
    nonmember_losses: Iterable[float],
    threshold: float | None = None,
) -> float:
    """[PARTIALLY_SPECIFIED] Table 1 MIR is a success rate; attack algorithm unnamed.

    Yeom-style: predict member if loss < threshold (default: mean of both).
    Returns accuracy as a percentage.
    """
    members = list(member_losses)
    nonmembers = list(nonmember_losses)
    if not members or not nonmembers:
        return 0.0
    if threshold is None:
        threshold = (sum(members) / len(members) + sum(nonmembers) / len(nonmembers)) / 2.0
    tp = sum(1 for x in members if x < threshold)
    tn = sum(1 for x in nonmembers if x >= threshold)
    return 100.0 * (tp + tn) / (len(members) + len(nonmembers))


@torch.no_grad()
def generate_private(
    model: PrivLLMGuard,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    epsilon_out: float,
    delta_u: float,
    monitor: RealTimePrivacyMonitor | None = None,
    apply_noise: bool = True,
    bos_id: int = 1,
    eos_id: int | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """§IV.B Eq. 7 — exponential-mechanism decoding with per-token ε_t = ε_out / T.

    §IV.D: for T=512, ε₀ = 0.1/512 ≈ 1.95e-4 when ε_out is the generation share.
    Set greedy=True to decode argmax (non-private baseline for the demo).
    """
    model.eval()
    device = input_ids.device
    noise = model.calibrate_noise(input_ids, apply_noise=apply_noise)
    memory, src_pad = model.encode(input_ids, noise, apply_noise=apply_noise)
    batch = input_ids.size(0)
    ys = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    eps_t = epsilon_out / max(max_new_tokens, 1)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        hidden = model.decode(ys, memory, src_pad, noise, apply_noise=apply_noise)
        logits = model.lm_head(hidden[:, -1, :])  # (batch, vocab)
        if greedy:
            next_tok = logits.argmax(dim=-1)
        else:
            next_tok = exponential_mechanism_sample(logits, eps_t, delta_u)
        if eos_id is not None:
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            finished = finished | (next_tok == eos_id)
        ys = torch.cat([ys, next_tok.unsqueeze(-1)], dim=1)
        if monitor is not None:
            leak = model.privacy_risk_assessor(hidden[:, -1:, :])
            monitor.tracker.consume(eps_t)
            if monitor.should_recalibrate(leak.mean(dim=-1)):
                noise = model.anc.emergency_recalibrate(noise)
        if bool(finished.all()):
            break
    return ys


@torch.no_grad()
def timed_forward(model: PrivLLMGuard, input_ids: torch.Tensor, repeats: int = 5) -> float:
    """§V Table 3 latency helper (milliseconds). Will not match A100 245 ms on CPU."""
    model.eval()
    # warmup
    _ = model(input_ids, apply_noise=True)
    if input_ids.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(input_ids, apply_noise=True)
        if input_ids.is_cuda:
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed * 1000.0


def make_generation_monitor(epsilon_out: float, window_size: int = 50) -> RealTimePrivacyMonitor:
    tracker = PrivacyBudgetTracker(total_budget=epsilon_out, window_size=window_size)
    return RealTimePrivacyMonitor(tracker)
