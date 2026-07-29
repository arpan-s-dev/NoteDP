"""Train a tiny CPU checkpoint so the Hugging Face demo can decode readable text.

This is a showcase trainer: higher LR, no DP-SGD, overfits the 7 synthetic
templates. Paper training is src/train.py (DP-SGD, Appendix A).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import SYNTHETIC_TEMPLATES, SyntheticClinicalNotes
from src.loss import LossWeights, combined_loss
from src.model import ModelConfig, PrivLLMGuard
from src.tokenizer import WordTokenizer
from src.train import build_optimizer, set_seed
from src.utils import load_config


def main() -> None:
    cfg = load_config(str(ROOT / "configs" / "demo.yaml"))
    set_seed(int(cfg["training"]["seed"]))
    tokenizer = WordTokenizer.from_texts(list(SYNTHETIC_TEMPLATES.values()))
    cfg["model"]["vocab_size"] = tokenizer.vocab_size
    cfg["model"]["pad_token_id"] = tokenizer.pad_id
    max_len = int(cfg["model"]["max_seq_len"])

    train_ds = SyntheticClinicalNotes(
        num_samples=70,
        max_len=max_len,
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        split="train",
    )
    loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    model = PrivLLMGuard(ModelConfig.from_dict(cfg))
    opt = build_optimizer(model, cfg)
    weights = LossWeights.from_dict(cfg)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    epochs = int(cfg["training"]["epochs"])
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            out = model(
                batch["input_ids"].to(device),
                decoder_input_ids=batch["decoder_input_ids"].to(device),
                labels=batch["labels"].to(device),
                apply_noise=False,
            )
            parts = combined_loss(
                out["logits"],
                batch["labels"].to(device),
                out["leak_probs"],
                out["entity_logits"],
                batch["entity_labels"].to(device),
                weights,
            )
            parts["loss"].backward()
            opt.step()
            total += float(parts["loss"].item())
            n += 1
        print(f"epoch {epoch + 1}/{epochs}  loss={total / max(n, 1):.4f}  vocab={tokenizer.vocab_size}")

    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "demo.pt"
    torch.save(
        {"model": model.state_dict(), "cfg": cfg, "tokenizer": tokenizer.to_dict()},
        path,
    )
    print("saved", path)


if __name__ == "__main__":
    main()
