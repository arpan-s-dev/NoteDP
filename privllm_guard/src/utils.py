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

