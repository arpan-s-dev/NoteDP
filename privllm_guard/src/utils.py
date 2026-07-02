"""
PrivLLM-Guard — shared tensor helpers.

Paper: https://doi.org/10.1038/s41598-026-45883-6
Only utilities used by more than one module live here.
"""

from __future__ import annotations

from typing import Any

import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config (configs/base.yaml or configs/walkthrough.yaml)."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def subsequent_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Causal mask for the autoregressive decoder (§III — decoder is autoregressive).

    Returns:
        mask: 1 = keep, 0 = block — shape: (1, 1, size, size)
    """
    # [UNSPECIFIED] Paper does not give the mask tensor layout.
    # Using: 1 = attend, 0 = masked (matches official pllm.py masked_fill(mask == 0, -inf))
    attn_shape = (1, 1, size, size)
    mask = torch.tril(torch.ones(attn_shape, device=device, dtype=torch.float32))
    return mask


def padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Key padding mask from pad ids.

    Args:
        input_ids: (batch, seq_len)
        pad_token_id: integer pad id

    Returns:
        mask: (batch, 1, 1, seq_len) with 1 = keep, 0 = pad
    """
    keep = (input_ids != pad_token_id).float()  # (batch, seq_len)
    return keep[:, None, None, :]  # (batch, 1, 1, seq_len)


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Elementwise AND of optional 0/1 masks, broadcasting as needed."""
    combined: torch.Tensor | None = None
    for mask in masks:
        if mask is None:
            continue
        combined = mask if combined is None else combined * mask
    return combined
