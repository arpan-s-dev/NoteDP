"""Local FastAPI UI: chart review + PrivLLM-Guard privacy panel."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.charts import CHARTS, get_chart, list_charts
from src.pipeline import DemoEngine
from src.redact import find_highlights, sanitize_chart

WEBUI = ROOT / "webui"
ENGINE: DemoEngine | None = None

app = FastAPI(title="NoteDP")
app.mount("/static", StaticFiles(directory=WEBUI), name="static")


def engine() -> DemoEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = DemoEngine()
    return ENGINE


class RunBody(BaseModel):
    chart_id: str
    epsilon: float = Field(1.0, ge=0.01, le=1.0)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEBUI / "index.html")


@app.get("/api/charts")
def api_charts() -> dict:
    return {
        "disclaimer": (
            "All charts are fictional. No real patients. Names, MRNs, phones, "
            "and towns are invented for an internship demo."
        ),
        "charts": list_charts(),
    }


@app.get("/api/charts/{chart_id}")
def api_chart(chart_id: str) -> dict:
    try:
        chart = get_chart(chart_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown synthetic chart") from exc
    return {
        **chart.__dict__,
        "sanitized": sanitize_chart(chart),
        "highlights": find_highlights(chart),
    }


@app.post("/api/run")
def api_run(body: RunBody) -> dict:
    try:
        chart = get_chart(body.chart_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown synthetic chart") from exc
    result = engine().run(chart.excerpt, float(body.epsilon))
    return {
        "chart_id": chart.id,
        "original_note": chart.note,
        "excerpt": chart.excerpt,
        "sanitized": sanitize_chart(chart),
        "highlights": find_highlights(chart),
        "non_private": result.non_private,
        "private": result.private,
        "profile": result.profile,
        "sigma_emb": result.sigma_emb,
        "sigma_att": result.sigma_att,
        "risk": result.risk,
        "epsilon": result.epsilon,
        "delta": result.delta,
        "epsilon_remaining": result.epsilon_remaining,
        "window_cost": result.window_cost,
        "recalibrated": result.recalibrated,
        "embedding_cosine": result.embedding_cosine,
        "bleu": result.bleu,
        "rouge": result.rouge,
        "latency_ms": result.latency_ms,
        "budget_split": result.budget_split,
        "risk_tag": chart.risk_tag,
        "display_name": chart.display_name,
        "specialty": chart.specialty,
    }
