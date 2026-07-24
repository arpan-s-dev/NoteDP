# Reproduction Notes: PrivLLM-Guard

> This document records every implementation choice, whether it was specified by the paper,
> and what alternatives exist. If you're reproducing this paper, **read this first.**

---

## Paper

- **Title:** An adaptive differential privacy framework for clinical llms with context-aware noise calibration, hierarchical budgeting, and real-time auditing
- **Authors:** Ans D. Alghamdi
- **Year:** 2026
- **DOI:** https://doi.org/10.1038/s41598-026-45883-6
- **Official code:** https://github.com/ansbuedu/code5 (`pllm.py`)

---

## What this implements

PrivLLM-Guard's four-module privacy stack: a privacy-aware transformer encoder (Eqs. 3–4), an autoregressive decoder with exponential-mechanism sampling (Eq. 7), sequence-level adaptive noise calibration, DP-SGD with adaptive clipping (Eqs. 5–6, 11), hierarchical (ε,δ) budgets with RDP accounting (Eqs. 9, 16–17), and a real-time sliding-window privacy monitor (Eqs. 14, 18–19). Training and evaluation run on **synthetic** clinical notes only.

---

## Verified against

- [x] Paper equations in Mathematical modeling (§IV, Eqs. 1–19 as numbered in the HTML body)
- [x] Algorithm 1 **prose** (figure itself is not machine-readable)
- [x] Official code used only for unspecified defaults (`n_heads`, `d_ff`, dropout, pre-norm, GELU, init, AdamW weight decay)
- [ ] Paper Table 2 BLEU/ROUGE numbers — require MIMIC-III/i2b2 + 8×A100, not claimed here
- [ ] Official `pllm.py` end-to-end — we **do not** copy its σ×0.01 DP-SGD hack or encoder-only architecture

---

## Unspecified choices

| Component | Our Choice | Alternatives | Paper Quote (if partial) | Section |
|-----------|-----------|--------------|--------------------------|---------|
| Attention heads | 12 | 8 (official README) | — | — |
| d_ff | 3072 | 2048 | — | — |
| Vocab size | 30522 | Clinical tokenizer | ClinicalBERT cited | §V |
| Tokenizer | Hash toy tokenizer in skeleton | ClinicalBERT WordPiece | — | — |
| Positional encoding | Learned embeddings | Sinusoidal | — | — |
| FFN activation | GELU | ReLU, SiLU | — | — |
| Pre/post LN | Pre-LN | Post-LN | — | — |
| LayerNorm ε | 1e-5 | 1e-6 | — | — |
| Dropout | 0.1 after attn/FFN/emb | 0.0 under DP | — | — |
| Decoder depth | = encoder L | Smaller decoder | “autoregressive decoders” | §III |
| ANC σ multipliers | high 1.5 / med 1.0 / low 0.5 | RL policy | profile → σ unspecified | §III |
| Per-layer σ_att | Linear 1.15→0.85 | Uniform | “varying privacy budgets” | §III |
| AdamW β, ε | (0.9, 0.999), 1e-8 | (0.9, 0.98), 1e-9 | “AdamW” only | Appendix A |
| Weight decay | 0.01 | 0.0 | — | — |
| LR schedule | Cosine over epochs | Linear warmup | — | — |
| Early-stop | Patience 2 on val loss | Unnamed | “early-stopped” | Appendix A |
| L_utility | Eq. 13 coherence CE | Token accuracy | named, not defined | Eq. 12 |
| L_privacy | Mean leak probability | Spent ε | named, not defined | Eq. 12 |
| Δu | 1.0 | Per-step logit range | “empirically tuned” | Appendix A |
| Exponential u(·) | LM logits | log-softmax | Eq. 7 | §IV.B |
| Recalibration | Bump profile toward high | Halt generation | “emergency recalibration” | §III |
| R(y) halt threshold | 0.5 | Unnamed “predefined” | Eq. 14 | §IV.D |
| BLEU implementation | Local 4-gram BLEU | sacrebleu | “BLEU-4” | §V |
| BOS token | id 1 | Tokenizer BOS | — | — |
| Seed | 42 | Paper: 5 unnamed seeds | §V | — |

---

## Known deviations

| Deviation | Paper says | We do | Reason |
|-----------|-----------|-------|--------|
| Equation numbering | Abstract: attention is Eq. 5, DP-SGD Eqs. 3–4 | Body numbering (attention Eq. 4, clip Eq. 5, Gaussian Eq. 6) | Method-section equations win |
| Hierarchical ε | §V: 0.025/0.035/0.020/0.020 | Same as §V | Official code uses 0.04/0.03/0.02/0.01 |
| DP-SGD σ | Eq. 6 as written | Eq. 6 as written | Official multiplies σ by 0.01 |
| Architecture | Encoder + autoregressive decoder | Encoder–decoder | Official is encoder-only + LM head |
| Coherence loss | Eq. 13 log-prob | Eq. 13 | Official used cosine similarity |
| ANC | Sets (σ_emb, σ_att) from profile | Analyzer drives multipliers | Official computes analyzer and ignores it |
| Embedding noise at eval | Inference privacy described | `apply_noise` flag | Official noise only if `self.training` |
| Data | MIMIC-III / i2b2 / proprietary | Synthetic notes | No real PHI in this scaffold; datasets are credentialed |
| Scale | d=768, L=12, n=512, 8×A100 | `configs/base.yaml` has paper scale; walkthrough uses tiny dims | CPU sanity checks |
| RL for α,β,γ | Eq. 10 “learned through RL” | Fixed Appendix A λ | RL algorithm unspecified |
| Table 2 utility | BLEU-4=0.897 at ε=0.1 | Not reproduced | Missing data/compute; treat as target only |

**Honest limitation:** BLEU-4 of 0.897 under ε=0.1 on millions of notes is an extraordinary claim. This repository implements the *stated mechanisms*. Matching Table 2 would require the authors' data, teacher, and training run.

---

## Expected results

| Metric | Paper's number | Dataset | Conditions |
|--------|---------------|---------|------------|
| BLEU-4 | 0.897 | MIMIC-III/i2b2/proprietary | Table 2, ε=0.1, δ=1e-6 |
| ROUGE-L | 0.923 | same | Table 2 |
| Entity accuracy | 91.3% | same | §V |
| MIR | 4.2% | attack eval | Table 1 |
| AIR | 2.8% | attack eval | Table 1 |
| MES | 5.7% | attack eval | Table 1 |
| DLS | 0.11 | attack eval | §V Privacy Score example |
| Privacy Score | 9.3 | derived | §V worked example |
| Latency | 245 ms | 512 tokens, A100 | Table 3 |
| Memory | 4.2 GB | inference | Table 3 |
| Throughput | 19.3 req/s | inference | Table 3 |
| RDP vs composition | ε_total=0.093 vs ≈0.214 | 512 tokens, ε₀=1.95e-4 | §IV.D |

**Sanity checks that *should* pass on CPU (`configs/walkthrough.yaml`):**
- Eq. 3 changes embeddings when σ_emb>0
- Eq. 4 attention scores differ with vs without N_att
- Eq. 5 never increases gradient L2
- Eq. 6 σ grows as ε shrinks
- Eq. 9 parts sum to ε_total
- Eq. 17 is tighter than naive kε composition for the paper's 512-token example
- Privacy Score formula reproduces 9.44 before rounding to 9.3

---

## Debugging tips

1. **NaNs after attention:** a fully padded query row + `-inf` mask → softmax NaN. We `nan_to_num` attention weights to 0.
2. **Loss explodes:** Eq. 6 noise without the unofficial 0.01 factor is *much* larger than `pllm.py`. Use walkthrough dims and few steps first; consider spending ε_dec over *steps* not epochs.
3. **Utility near zero under DP:** expected at ε=0.1 on a tiny random-init model. Paper utility assumes a trained ClinicalBERT-scale network plus distillation.
4. **RDP ε > 0.1 immediately:** you are accumulating Gaussian RDP with sensitivity=C and large σ incorrectly, or calling `accumulate` per parameter instead of per step.
5. **Official code mismatch:** README architecture ≠ `PLLMConfig` ≠ paper §V. Trust Appendix A + §V, not the README table.

---

## Scope decisions

### Implemented
- Privacy-aware encoder/decoder, ANC, DP-SGD microbatch, RDP, monitor, exponential mechanism, distillation hook, synthetic data, metric functions — core contribution

### Intentionally excluded
- Baselines (DP-GPT-3.5, PrivClinicalT5, …)
- Official figure dashboard (`fig4_*.png` generators)
- 8-GPU training, CUDA noise kernels, model compression
- FHIR/EHR integration, clinician UI
- Real MIMIC/i2b2 download

### Needed for full reproduction (not included)
- PhysioNet credentialed MIMIC-III (not the Kaggle snapshot cited in the paper)
- i2b2 NLP challenge license
- Non-private teacher trained on the same corpus
- 8×A100, 5 random seeds, specialty-stratified 70/15/15 on encounter IDs
- Attack implementations that produced Table 1 (unnamed)

---

## References

- Vaswani et al., 2017 — transformer residual/attention skeleton
- Huang et al., ClinicalBERT (arXiv:1904.05342) — d=768, L=12, n=512 cited in §V
- Abadi et al., 2016 CCS — DP-SGD; paper cites this as reference 37 for the accountant
- Mironov, Rényi DP — Eqs. 16–17 composition
- Official `pllm.py` — unspecified architecture/training defaults only
