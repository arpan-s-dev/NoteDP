"""Gradio UI for the PrivLLM-Guard internship demo."""

from __future__ import annotations

import gradio as gr

from src.charts import CHARTS
from src.pipeline import DemoEngine

ENGINE: DemoEngine | None = None

PAPER = (
    "**Paper:** Alghamdi, *An adaptive differential privacy framework for clinical LLMs…*, "
    "Scientific Reports 16:15781 (2026). System name: **PrivLLM-Guard**.\n\n"
    "**Internship demo (CPU toy model).** This is not the paper’s 768-d, 12-layer, "
    "MIMIC-III run. It shows the *mechanisms*: adaptive noise, hierarchical ε, "
    "and private decoding on **synthetic** notes only. Do not paste real patient records."
)


def get_engine() -> DemoEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = DemoEngine()
    return ENGINE


def run_demo(note: str, epsilon: float) -> tuple:
    text = (note or "").strip()
    if not text:
        raise gr.Error("Paste or pick a synthetic clinical note first.")
    lowered = text.lower()
    banned = ["ssn", "social security", "mrn:", "date of birth"]
    if any(k in lowered for k in banned):
        raise gr.Error("This demo only accepts synthetic notes. Do not paste real identifiers.")
    result = get_engine().run(text, float(epsilon), use_private_decoding=True)
    metrics = (
        f"**ANC profile:** `{result.profile}`  \n"
        f"**σ_emb / σ_att:** `{result.sigma_emb:.3f}` / `{result.sigma_att:.3f}`  \n"
        f"**Leak risk R(y) (Eq. 14, mean):** `{result.risk:.3f}`  \n"
        f"**ε / δ:** `{result.epsilon}` / `{result.delta:.0e}`  \n"
        f"**ε remaining (output share):** `{result.epsilon_remaining:.4f}`  \n"
        f"**Window cost:** `{result.window_cost:.4f}`  \n"
        f"**Emergency recalibration:** `{result.recalibrated}`  \n"
        f"**Embedding cosine (clean vs Eq. 3 noise):** `{result.embedding_cosine:.3f}`  \n"
        f"**BLEU-4 / ROUGE-L (private vs greedy):** `{result.bleu:.3f}` / `{result.rouge:.3f}`  \n"
        f"**Latency:** `{result.latency_ms:.0f} ms` (CPU demo, not Table 3’s 245 ms on A100)"
    )
    budget_rows = [[k, f"{v:.4f}"] for k, v in result.budget_split.items()]
    return result.original, result.non_private, result.private, metrics, budget_rows


def build_demo() -> gr.Blocks:
    examples = [[c.excerpt, 1.0] for c in CHARTS]
    with gr.Blocks(title="ChartCloak internship demo") as demo:
        gr.Markdown("# ChartCloak\n" + PAPER)
        with gr.Tab("Live demo"):
            note = gr.Textbox(
                label="Synthetic clinical note",
                lines=6,
                placeholder="Choose an example below. Synthetic text only.",
            )
            epsilon = gr.Slider(
                0.01, 1.0, value=1.0, step=0.01,
                label="Privacy budget ε (paper default is 0.1; this tiny CPU model needs larger ε to stay readable)",
            )
            go = gr.Button("Run ChartCloak", variant="primary")
            with gr.Row():
                original = gr.Textbox(label="Original synthetic note", lines=5)
                nonpriv = gr.Textbox(label="Reconstruction without DP noise", lines=5)
                priv = gr.Textbox(label="Private reconstruction (Eq. 3–4 noise)", lines=5)
            metrics = gr.Markdown()
            budget = gr.Dataframe(
                headers=["Budget part (Eq. 9)", "ε"],
                label="Hierarchical privacy budget",
            )
            go.click(
                run_demo,
                inputs=[note, epsilon],
                outputs=[original, nonpriv, priv, metrics, budget],
            )
            gr.Examples(examples=examples, inputs=[note, epsilon], label="Synthetic examples")
        with gr.Tab("What this is"):
            gr.Markdown(
                """
## Elevator pitch

Hospitals want LLMs for summarization. Notes can identify patients. **ChartCloak**
runs **PrivLLM-Guard**
adds calibrated Gaussian noise and a tracked privacy budget so one patient’s record
cannot change the model’s behavior much.

## What you are looking at

| Column | Meaning |
|---|---|
| Reconstruction without DP | Same tiny model, no privacy noise — utility baseline |
| Private reconstruction | Embedding + attention noise (Eqs. 3–4), then argmax |
| ANC profile | Once-per-sequence high / medium / low sensitivity |
| Hierarchical ε | Encoder, decoder, attention, output shares from §V |

## Hosting

**Local / Gradio share** works on a laptop. **Hugging Face Gradio Spaces currently need PRO**. **Vercel cannot** run the PyTorch model; `web/` is only an iframe shell.

## Not claimed

Paper BLEU-4 = 0.897, A100 latency, or HIPAA certification. This demo is a
mechanism showcase on synthetic notes.
"""
            )
    return demo
