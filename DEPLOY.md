# How to run or host this demo

The chart UI is a **local FastAPI app**. That is the intended demo. Hugging Face Gradio Spaces currently return HTTP 402 on free CPU (PRO required). Vercel cannot run PyTorch.

## 1. On this machine (recommended)

```powershell
cd <repo-root>
python -m pip install -r requirements.txt
copy .env.example .env   # optional; defaults are already 127.0.0.1:8080
python app.py
```

Open http://127.0.0.1:8080

Optional env vars from `.env.example`: `PRIVLLM_HOST`, `PRIVLLM_PORT`.

## 2. Optional Gradio UI (same laptop)

```powershell
cd privllm_guard
python app.py
```

`gradio.launch(share=True)` can try a `*.gradio.live` tunnel. That often fails depending on the network. Prefer the FastAPI URL above.

## 3. Hugging Face Spaces (PRO)

1. https://huggingface.co/new-space
2. SDK: Gradio, hardware: CPU basic
3. Point `app_file` at `privllm_guard/app.py` (Gradio), not the root FastAPI `app.py`
4. Free CPU currently requires PRO (HTTP 402 without it)

## 4. Vercel (landing page only)

`web/` is a static iframe shell. It does **not** run the encoder–decoder. Do not expect a live model there.
