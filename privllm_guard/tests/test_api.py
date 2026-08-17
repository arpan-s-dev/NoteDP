from fastapi.testclient import TestClient

from src.server import app

client = TestClient(app)


def test_index_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert b"NoteDP" in res.content


def test_charts_api():
    res = client.get("/api/charts")
    assert res.status_code == 200
    body = res.json()
    assert len(body["charts"]) == 10
    assert "fictional" in body["disclaimer"].lower()


def test_one_chart_has_sanitized_note():
    res = client.get("/api/charts/syn-4401")
    assert res.status_code == 200
    body = res.json()
    assert "Elena Voss" in body["note"]
    assert "<PATIENT>" in body["sanitized"]
