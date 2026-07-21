"""Internship demo pipeline: one synthetic note in, privacy dashboard + two summaries out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F

from src.data import SYNTHETIC_TEMPLATES
from src.evaluate import bleu4, make_generation_monitor, rouge_l
from src.model import ModelConfig, PrivLLMGuard
from src.privacy import GaussianMechanism
from src.tokenizer import WordTokenizer
from src.utils import load_config


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "demo.pt"


@dataclass
class DemoResult:
    original: str
    non_private: str
    private: str
    profile: str
    sigma_emb: float
    sigma_att: float
    risk: float
    epsilon: float
    delta: float
    epsilon_remaining: float
    window_cost: float
    recalibrated: bool
    embedding_cosine: float
    bleu: float
    rouge: float
    latency_ms: float
    budget_split: dict[str, float]


def pad_ids(ids: list[int], max_len: int, pad_id: int) -> torch.Tensor:
    ids = ids[:max_len]
    ids = ids + [pad_id] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


class DemoEngine:
    def __init__(self, checkpoint: Path | None = None) -> None:
        self.device = torch.device("cpu")
        ckpt_path = checkpoint or CHECKPOINT
        if ckpt_path.exists():
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.cfg = blob["cfg"]
            self.tokenizer = WordTokenizer(blob["tokenizer"])
            self.cfg["model"]["vocab_size"] = self.tokenizer.vocab_size
            self.cfg["model"]["pad_token_id"] = self.tokenizer.pad_id
            self.model = PrivLLMGuard(ModelConfig.from_dict(self.cfg))
            self.model.load_state_dict(blob["model"])
        else:
            self.cfg = load_config(str(ROOT / "configs" / "demo.yaml"))
            self.tokenizer = WordTokenizer.from_texts(list(SYNTHETIC_TEMPLATES.values()))
            self.cfg["model"]["vocab_size"] = self.tokenizer.vocab_size
            self.cfg["model"]["pad_token_id"] = self.tokenizer.pad_id
            self.model = PrivLLMGuard(ModelConfig.from_dict(self.cfg))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def run(self, text: str, epsilon: float, use_private_decoding: bool = True) -> DemoResult:
        import time

        cfg = self.cfg
        tok = self.tokenizer
        max_len = int(cfg["model"]["max_seq_len"])
        ids = tok.encode(text)
        src = pad_ids(ids, max_len, tok.pad_id).to(self.device)
        valid = min(len(ids), max_len)

        # Scale paper σ with the requested ε (smaller ε → more noise).
        base_eps = float(cfg["privacy"]["epsilon_total"])
        scale = base_eps / max(epsilon, 1e-4)
        self.model.anc.base_sigma_emb = float(cfg["privacy"]["sigma_emb"]) * scale
        self.model.anc.base_sigma_att = float(cfg["privacy"]["sigma_att"]) * scale

        t0 = time.perf_counter()
        # Teacher-forced reconstruction matches how the demo checkpoint was trained.
        # Autoregressive-from-BOS is still in evaluate.generate_private (Eq. 7).
        clean_out = self.model(src, decoder_input_ids=src, apply_noise=False)
        priv_out = self.model(src, decoder_input_ids=src, apply_noise=True)
        noise = priv_out["noise"]
        leak = priv_out["leak_probs"][:, :valid]
        risk = float(leak.mean().item())

        clean_ids = clean_out["logits"].argmax(dim=-1)[0, :valid].tolist()
        # Private text: Eq. 3–4 noise then argmax. Per-token Eq. 7 sampling at
        # ε_out/T is near-uniform on this tiny model, so the live demo uses
        # noisy argmax to show the privacy–utility tradeoff. Eq. 7 remains in
        # evaluate.generate_private.
        priv_ids = priv_out["logits"].argmax(dim=-1)[0, :valid].tolist()
        eps_out = float(cfg["privacy"]["epsilon_out"]) * (epsilon / base_eps)
        monitor = make_generation_monitor(eps_out, window_size=int(cfg["privacy"]["window_size"]))
        monitor.tracker.consume(eps_out)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        clean_emb = self.model.embedding(src, sigma_emb=0.0, apply_noise=False)
        noisy_emb = GaussianMechanism.add_noise(clean_emb, noise.sigma_emb)
        cosine = float(
            F.cosine_similarity(clean_emb.flatten(), noisy_emb.flatten(), dim=0).item()
        )
        return DemoResult(
            original=text.strip(),
            non_private=tok.decode(clean_ids),
            private=tok.decode(priv_ids),
            profile=noise.profile,
            sigma_emb=noise.sigma_emb,
            sigma_att=noise.sigma_att,
            risk=risk,
            epsilon=epsilon,
            delta=float(cfg["privacy"]["delta"]),
            epsilon_remaining=monitor.tracker.remaining,
            window_cost=monitor.tracker.window_cost,
            recalibrated=monitor.should_recalibrate(),
            embedding_cosine=cosine,
            bleu=bleu4(priv_ids, clean_ids),
            rouge=rouge_l(priv_ids, clean_ids),
            latency_ms=latency_ms,
            budget_split={
                "encoder ε": float(cfg["privacy"]["epsilon_enc"]) * (epsilon / base_eps),
                "decoder ε": float(cfg["privacy"]["epsilon_dec"]) * (epsilon / base_eps),
                "attention ε": float(cfg["privacy"]["epsilon_att"]) * (epsilon / base_eps),
                "output ε": eps_out,
            },
        )
