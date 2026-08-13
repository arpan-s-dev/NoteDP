from src.charts import CHARTS
from src.redact import find_highlights


def test_highlights_include_name_and_mrn():
    chart = CHARTS[0]
    texts = {h["text"] for h in find_highlights(chart)}
    assert chart.display_name in texts
    assert chart.mrn in texts
