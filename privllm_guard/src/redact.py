"""Display-only generalization for fictional charts (not a certified de-identifier)."""

from __future__ import annotations

import re

from src.charts import SyntheticChart


PHONE_RE = re.compile(r"555-\d{3}-\d{4}")
EMAIL_RE = re.compile(r"[\w.+-]+@example\.invalid")
DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?\b"
)


def age_band(age: int) -> str:
    decade = (age // 10) * 10
    return f"patient in their {decade}s"


def sanitize_chart(chart: SyntheticChart) -> str:
    """Risk-adaptive display: keep clinical facts, generalize identifiers."""
    text = chart.note
    text = text.replace(chart.display_name, "<PATIENT>")
    text = text.replace(chart.mrn, "<MRN>")
    text = PHONE_RE.sub("<PHONE>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = DATE_RE.sub("<DATE>", text)
    text = re.sub(
        rf"\b{chart.age}-year-old\b",
        age_band(chart.age),
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"\b{chart.age} year old\b",
        age_band(chart.age),
        text,
        flags=re.I,
    )
    if chart.risk_tag == "high":
        text = re.sub(
            r"retired neurosurgeon|neurosurgeon|night pharmacist|chemistry teacher|long-haul truck driver",
            "<OCCUPATION>",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"Graymere Crossing|Red Kettle, NM|Cedar Hollow|Lake Meridian|North Quay|Port Sable",
            "<REGION>",
            text,
        )
    return text


def find_highlights(chart: SyntheticChart) -> list[dict[str, str]]:
    """Spans to paint in the original note (exact string matches)."""
    needles = [
        ("name", chart.display_name),
        ("mrn", chart.mrn),
        ("date", chart.encounter_date),
    ]
    found: list[dict[str, str]] = []
    for kind, needle in needles:
        if needle and needle in chart.note:
            found.append({"kind": kind, "text": needle})
    for m in PHONE_RE.finditer(chart.note):
        found.append({"kind": "phone", "text": m.group(0)})
    for m in EMAIL_RE.finditer(chart.note):
        found.append({"kind": "email", "text": m.group(0)})
    found.append({"kind": "age", "text": f"{chart.age}-year-old"})
    found.append({"kind": "age", "text": f"{chart.age} year old"})
    return found
