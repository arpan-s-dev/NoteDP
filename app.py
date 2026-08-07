"""Local entry: clinical chart UI (FastAPI), not Gradio."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "privllm_guard"
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("PRIVLLM_HOST", "127.0.0.1")
    port = int(os.environ.get("PRIVLLM_PORT", "8080"))
    print(f"ChartCloak local UI -> http://{host}:{port}")
    print("All charts are fictional. Do not paste real patient records.")
    uvicorn.run("src.server:app", host=host, port=port, reload=False)
