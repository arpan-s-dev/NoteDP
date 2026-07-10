"""
PrivLLM-Guard — dataset skeleton.

Paper: https://doi.org/10.1038/s41598-026-45883-6  §V Experimental setup

The paper trains on MIMIC-III, i2b2, and a proprietary hospital corpus.
Those datasets are NOT downloaded here (credentialed / identifiable).

This module:
  1. Builds fully synthetic clinical-style notes (no real patients).
  2. Documents how a user would window real de-identified notes (512 / 128).
  3. Stratifies 70/15/15 by specialty with no encounter leakage (§V).

TODO: Replace SyntheticClinicalNotes with a loader over locally authorized
de-identified files. Do not commit real PHI into this repository.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

from src.tokenizer import WordTokenizer
from src.charts import CHARTS

SPECIALTIES = [
    "cardiology",
    "oncology",
    "neurology",
    "emergency_medicine",
    "internal_medicine",
    "radiology",
    "pathology",
]

# [FROM_OFFICIAL_CODE] 7 entity types used by the entity head.
ENTITY_TYPES = [
    "diagnosis",
    "procedure",
    "medication",
    "lab_value",
    "anatomy",
    "symptom",
    "other",
]

# Eq. 14 w_i — [PARTIALLY_SPECIFIED] "derived from NER and ICD codes".
# Higher weight = more identifying. Identifiers/dates > clinical terms,
# matching §V "60% of noise to high-sensitivity tokens … 15% to ICD/meds".
ENTITY_WEIGHTS = {
    "diagnosis": 0.4,
    "procedure": 0.3,
    "medication": 0.25,
    "lab_value": 0.2,
    "anatomy": 0.15,
    "symptom": 0.2,
    "other": 1.0,  # catch-all includes names/dates in the synthetic scheme
}

SYNTHETIC_TEMPLATES = {
    chart.specialty.lower().replace(" ", "_").split("/")[0].strip(): chart.excerpt
    for chart in CHARTS
}
SYNTHETIC_TEMPLATES.update({chart.id: chart.excerpt for chart in CHARTS})


@dataclass
class WindowingSpec:
    """§V — 512-token windows with 128-token overlap; per-window RDP."""

    window_size: int = 512
    overlap: int = 128


def sliding_windows(token_ids: list[int], spec: WindowingSpec) -> Iterator[list[int]]:
    """Yield overlapping windows. Short sequences yield a single padded-later window."""
    step = spec.window_size - spec.overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than window_size")
    if len(token_ids) <= spec.window_size:
        yield token_ids
        return
    start = 0
    while start < len(token_ids):
        yield token_ids[start : start + spec.window_size]
        if start + spec.window_size >= len(token_ids):
            break
        start += step


class SyntheticClinicalNotes(Dataset):
    """Synthetic notes only — no real identifiable records (§ user constraint / paper ethics).

    Tokenization is a character-hash toy tokenizer so the scaffold runs without
    downloading ClinicalBERT. Swap in AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    when you have authorized weights.
    """

    def __init__(
        self,
        num_samples: int,
        max_len: int,
        vocab_size: int,
        specialties: list[str] | None = None,
        seed: int = 42,
        split: str = "train",
        tokenizer: WordTokenizer | None = None,
    ) -> None:
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.specialties = specialties or SPECIALTIES
        self.split = split
        self.tokenizer = tokenizer
        rng = random.Random(seed + {"train": 0, "val": 1, "test": 2}.get(split, 0))
        self.records: list[dict[str, Any]] = []
        for i in range(num_samples):
            specialty = self.specialties[i % len(self.specialties)]
            text = SYNTHETIC_TEMPLATES[specialty]
            # encounter_id groups windows from one note — §V no leakage across splits
            encounter_id = f"{split}-{i}"
            if tokenizer is not None:
                tokens = tokenizer.encode(text)[:max_len]
            else:
                tokens = self._hash_tokenize(text, rng)
            entity = [
                rng.randrange(len(ENTITY_TYPES)) if tok > 10 else 6 for tok in tokens
            ]
            self.records.append(
                {
                    "input_ids": tokens,
                    "entity_labels": entity,
                    "specialty": specialty,
                    "encounter_id": encounter_id,
                    "text": text,
                }
            )

    def _hash_tokenize(self, text: str, rng: random.Random) -> list[int]:
        # [UNSPECIFIED] Paper uses ClinicalBERT tokenization; this is a scaffold.
        ids = [1]  # fake CLS
        for word in text.lower().split():
            ids.append(2 + (hash(word) % (self.vocab_size - 12)))
        ids.append(2)  # fake SEP
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        return ids

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        tokens = rec["input_ids"]
        entity = rec["entity_labels"]
        seq = torch.tensor(tokens, dtype=torch.long)
        ent = torch.tensor(entity, dtype=torch.long)
        pad_len = self.max_len - seq.size(0)
        if pad_len > 0:
            seq = torch.nn.functional.pad(seq, (0, pad_len), value=0)
            labels = torch.nn.functional.pad(
                torch.tensor(tokens, dtype=torch.long), (0, pad_len), value=-100
            )
            ent = torch.nn.functional.pad(ent, (0, pad_len), value=-100)
        else:
            labels = seq.clone()
        weights = torch.tensor(
            [
                ENTITY_WEIGHTS[ENTITY_TYPES[int(e)]] if int(e) >= 0 else 0.0
                for e in ent.tolist()
            ],
            dtype=torch.float32,
        )
        return {
            "input_ids": seq,
            "labels": labels,
            "decoder_input_ids": seq.clone(),
            "entity_labels": ent,
            "token_weights": weights,
            "specialty": rec["specialty"],
            "encounter_id": rec["encounter_id"],
            "text": rec["text"],
        }


def stratified_splits(
    num_samples: int,
    max_len: int,
    vocab_size: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[SyntheticClinicalNotes, SyntheticClinicalNotes, SyntheticClinicalNotes]:
    """§V — 70/15/15. Encounter integrity is automatic (one note = one encounter)."""
    n_train = int(num_samples * train_frac)
    n_val = int(num_samples * val_frac)
    n_test = num_samples - n_train - n_val
    train = SyntheticClinicalNotes(n_train, max_len, vocab_size, seed=seed, split="train")
    val = SyntheticClinicalNotes(n_val, max_len, vocab_size, seed=seed, split="val")
    test = SyntheticClinicalNotes(n_test, max_len, vocab_size, seed=seed, split="test")
    return train, val, test
