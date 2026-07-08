"""
PrivLLM-Guard — model architecture.

Paper: https://doi.org/10.1038/s41598-026-45883-6
Alghamdi, Sci Rep 16:15781 (2026)

Implements:
  §III  Privacy-Aware Encoder + Differentially-Private Decoder
  §IV.A Eq. 3  embedding perturbation
  §IV.A Eq. 4  privacy-aware attention
  §III  hierarchical per-layer attention noise
  §III  sequence-level Adaptive Noise Calibration
  §IV.D Eq. 14 token leak probabilities from the risk head

Usage:
    from src.model import PrivLLMGuard, ModelConfig
    model = PrivLLMGuard(ModelConfig())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.privacy import (
    AdaptiveNoiseCalibrator,
    GaussianMechanism,
    NoiseParameters,
    hierarchical_layer_sigmas,
)
from src.utils import combine_masks, padding_mask, subsequent_mask


@dataclass
class ModelConfig:
    """Architecture hyperparameters.

    Defaults match §V (d=768, L=12, n=512) plus [FROM_OFFICIAL_CODE] fields.
    """

    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    n_decoder_layers: int = 12
    d_ff: int = 3072
    vocab_size: int = 30522
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    norm_eps: float = 1e-5
    pad_token_id: int = 0
    n_entity_types: int = 7
    sigma_emb: float = 0.5
    sigma_att: float = 0.3
    anc_high: float = 1.5
    anc_medium: float = 1.0
    anc_low: float = 0.5

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ModelConfig":
        model = cfg["model"]
        privacy = cfg["privacy"]
        data = cfg.get("data", {})
        anc = privacy.get("anc_sigma_multipliers", {})
        return cls(
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            n_layers=model["n_layers"],
            n_decoder_layers=model.get("n_decoder_layers", model["n_layers"]),
            d_ff=model["d_ff"],
            vocab_size=model["vocab_size"],
            max_seq_len=model["max_seq_len"],
            dropout=model["dropout"],
            activation=model.get("activation", "gelu"),
            norm_eps=float(model.get("norm_eps", 1e-5)),
            pad_token_id=int(model.get("pad_token_id", 0)),
            n_entity_types=int(data.get("n_entity_types", 7)),
            sigma_emb=float(privacy["sigma_emb"]),
            sigma_att=float(privacy["sigma_att"]),
            anc_high=float(anc.get("high", 1.5)),
            anc_medium=float(anc.get("medium", 1.0)),
            anc_low=float(anc.get("low", 0.5)),
        )


def _activation(name: str) -> nn.Module:
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unknown activation {name}")


class PrivacyAwareEmbedding(nn.Module):
    """§IV.A, Eq. 3 — ẽ_i = e_i + N(0, σ_emb² I) before hierarchical attention.

    "The encoder embedding transformation applies privacy-preserving
    perturbations to input representations before they enter the
    hierarchical attention layers."
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # [FROM_OFFICIAL_CODE] learned positional embeddings (paper unspecified)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        sigma_emb: float = 0.0,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            sigma_emb: cached ANC scale for this sequence
            apply_noise: whether to apply Eq. 3

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch, seq_len)  # (batch, seq_len)
        token_e = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        pos_e = self.position_embedding(positions)  # (batch, seq_len, d_model)
        e = token_e + pos_e  # (batch, seq_len, d_model)
        if apply_noise and sigma_emb > 0.0:
            # Eq. 3 — noise is applied even at inference when apply_noise=True.
            # Official code gates on self.training; the paper describes inference
            # privacy as well, so we honor the apply_noise flag instead.
            e = GaussianMechanism.add_noise(e, sigma_emb)
        return self.dropout(self.layer_norm(e))  # (batch, seq_len, d_model)


class PrivacyAwareAttention(nn.Module):
    """§IV.A, Eq. 4 — Attention(Q,K,V) = softmax( (QK^T + N_att) / √d_k ) V

    "We modify the standard attention mechanism to incorporate privacy noise,
    ensuring that attention weights … do not leak confidential information."
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads  # h in paper notation: n_heads
        self.d_model = config.d_model
        self.W_q = nn.Linear(config.d_model, config.d_model)
        self.W_k = nn.Linear(config.d_model, config.d_model)
        self.W_v = nn.Linear(config.d_model, config.d_model)
        self.W_o = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        sigma_att: float = 0.0,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (batch, seq_q/k, d_model)
            mask: 1=keep, 0=block, broadcastable to (batch, heads, seq_q, seq_k)
            sigma_att: per-layer ANC scale

        Returns:
            (batch, seq_q, d_model)
        """
        batch, seq_q, _ = query.shape
        seq_k = key.size(1)
        q = self.W_q(query)  # (batch, seq_q, d_model)
        k = self.W_k(key)  # (batch, seq_k, d_model)
        v = self.W_v(value)  # (batch, seq_k, d_model)
        q = q.view(batch, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        # (batch, n_heads, seq_*, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # (batch, n_heads, seq_q, seq_k)
        if apply_noise and sigma_att > 0.0:
            # Eq. 4 — N_att is added to the scores, then scaled? The displayed
            # equation is softmax( (QK^T + N_att) / √d_k ) V, so noise is
            # inside the parentheses before the √d_k divide. Official code
            # adds noise AFTER dividing by √d_k. We follow Eq. 4:
            #   scores = (QK^T)/√d_k already computed; equivalent to adding
            #   N_att/√d_k to these scores.
            scores = scores + GaussianMechanism.add_noise(
                torch.zeros_like(scores), sigma_att
            ) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_q, seq_k)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_q, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.W_o(out)  # (batch, seq_q, d_model)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with privacy-aware attention.

    [FROM_OFFICIAL_CODE] Pre-LN residual: x + sublayer(LN(x))
    [UNSPECIFIED] Paper does not state pre vs post norm.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = PrivacyAwareAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            _activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        sigma_att: float = 0.0,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        h = self.norm1(x)  # (batch, seq, d_model)
        x = x + self.self_attn(
            h, h, h, mask=mask, sigma_att=sigma_att, apply_noise=apply_noise
        )
        x = x + self.ff(self.norm2(x))
        return x  # (batch, seq, d_model)


class DecoderBlock(nn.Module):
    """§III — autoregressive decoder block with causal self-attention + cross-attention.

    Cross-attention is not given an extra equation; we reuse Eq. 4 noise on
    both self- and cross-attention. [PARTIALLY_SPECIFIED]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = PrivacyAwareAttention(config)
        self.cross_attn = PrivacyAwareAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            _activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        sigma_att: float = 0.0,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(
            h, h, h, mask=self_mask, sigma_att=sigma_att, apply_noise=apply_noise
        )
        h = self.norm2(x)
        x = x + self.cross_attn(
            h,
            memory,
            memory,
            mask=memory_mask,
            sigma_att=sigma_att,
            apply_noise=apply_noise,
        )
        x = x + self.ff(self.norm3(x))
        return x


class SensitivityAnalyzer(nn.Module):
    """§III — once-per-sequence high/medium/low privacy profile classifier.

    "At sequence start, a single forward pass through the sensitivity analyzer
    classifies the input's privacy profile (high/medium/low)."
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # [FROM_OFFICIAL_CODE] 2-layer MLP on mean-pooled states → 3 classes
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 3),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)  # (batch, d_model)
        return F.softmax(self.classifier(pooled), dim=-1)  # (batch, 3)


class PrivacyRiskAssessor(nn.Module):
    """§IV.D, Eq. 14 — token-level P(leak_i | y_{≤i}).

    w_i comes from NER/ICD weights supplied by the caller, not this module.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.risk_scorer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.risk_scorer(hidden_states).squeeze(-1)  # (batch, seq_len)


class PrivLLMGuard(nn.Module):
    """§III–§IV — PrivLLM-Guard encoder–decoder with ANC and risk head.

    Composed of:
      - PrivacyAwareEmbedding (Eq. 3)
      - Encoder stack of TransformerBlock (Eq. 4, hierarchical σ_att)
      - Decoder stack of DecoderBlock (autoregressive)
      - SensitivityAnalyzer (ANC, once per sequence)
      - PrivacyRiskAssessor (Eq. 14)
      - LM head + medical entity head (§IV.C L_medical)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = PrivacyAwareEmbedding(config)
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_decoder_layers)]
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.entity_classifier = nn.Linear(config.d_model, config.n_entity_types)
        self.sensitivity_analyzer = SensitivityAnalyzer(config)
        self.privacy_risk_assessor = PrivacyRiskAssessor(config)
        self.anc = AdaptiveNoiseCalibrator(
            sigma_emb=config.sigma_emb,
            sigma_att=config.sigma_att,
            multipliers={
                "high": config.anc_high,
                "medium": config.anc_medium,
                "low": config.anc_low,
            },
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # [FROM_OFFICIAL_CODE] Xavier uniform Linear; Embedding N(0, 0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        input_ids: torch.Tensor,
        noise: NoiseParameters,
        apply_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional encoder (§III).

        Returns:
            memory: (batch, src_len, d_model)
            src_pad_mask: (batch, 1, 1, src_len)
        """
        src_pad = padding_mask(input_ids, self.config.pad_token_id)
        hidden = self.embedding(
            input_ids, sigma_emb=noise.sigma_emb, apply_noise=apply_noise
        )
        layer_sigmas = hierarchical_layer_sigmas(noise.sigma_att, len(self.encoder_layers))
        for layer, sigma in zip(self.encoder_layers, layer_sigmas):
            hidden = layer(hidden, mask=src_pad, sigma_att=sigma, apply_noise=apply_noise)
        return hidden, src_pad

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        noise: NoiseParameters,
        apply_noise: bool = True,
    ) -> torch.Tensor:
        """Autoregressive decoder (§III)."""
        batch, tgt_len = tgt_ids.shape
        causal = subsequent_mask(tgt_len, device=tgt_ids.device)
        tgt_pad = padding_mask(tgt_ids, self.config.pad_token_id)
        self_mask = combine_masks(causal, tgt_pad)
        hidden = self.embedding(
            tgt_ids, sigma_emb=noise.sigma_emb, apply_noise=apply_noise
        )
        layer_sigmas = hierarchical_layer_sigmas(
            noise.sigma_att, len(self.decoder_layers)
        )
        for layer, sigma in zip(self.decoder_layers, layer_sigmas):
            hidden = layer(
                hidden,
                memory,
                self_mask=self_mask,
                memory_mask=memory_mask,
                sigma_att=sigma,
                apply_noise=apply_noise,
            )
        return hidden  # (batch, tgt_len, d_model)

    def calibrate_noise(
        self,
        input_ids: torch.Tensor,
        apply_noise: bool = True,
    ) -> NoiseParameters:
        """§III — ANC once per sequence from a noise-free probe pass."""
        with torch.no_grad():
            probe = self.embedding(input_ids, sigma_emb=0.0, apply_noise=False)
            probs = self.sensitivity_analyzer(probe)
        return self.anc.from_probs(probs)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        apply_noise: bool = True,
        noise: NoiseParameters | None = None,
    ) -> dict[str, torch.Tensor | NoiseParameters | None]:
        """Teacher-forced generation forward.

        If decoder_input_ids is omitted, we use shifted input_ids as a
        denoising/summarization stand-in (same document reconstruction).
        """
        if noise is None:
            noise = self.calibrate_noise(input_ids, apply_noise=apply_noise)
        memory, src_pad = self.encode(input_ids, noise, apply_noise=apply_noise)
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        dec_hidden = self.decode(
            decoder_input_ids, memory, src_pad, noise, apply_noise=apply_noise
        )
        logits = self.lm_head(dec_hidden)  # (batch, tgt_len, vocab)
        entity_logits = self.entity_classifier(memory)  # (batch, src_len, n_entities)
        leak_probs = self.privacy_risk_assessor(dec_hidden)  # (batch, tgt_len)
        # ANC probe is also run under no_grad in calibrate_noise. Official
        # pllm.py detaches here; the 3-way profile is not in Eq. 12, so it
        # does not receive LM gradients unless a profile label is supplied.
        sensitivity_scores = self.sensitivity_analyzer(memory.detach())
        return {
            "logits": logits,
            "entity_logits": entity_logits,
            "hidden_states": dec_hidden,
            "encoder_hidden": memory,
            "leak_probs": leak_probs,
            "sensitivity_scores": sensitivity_scores,
            "noise": noise,
            "labels": labels,
        }
