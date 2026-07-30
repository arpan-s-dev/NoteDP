"""Hugging Face Space / local Gradio entry (run from privllm_guard/)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_ui import build_demo

demo = build_demo()

if __name__ == "__main__":
    import os
    demo.launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
