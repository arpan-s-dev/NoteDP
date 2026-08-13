from src.charts import CHARTS, get_chart
from src.redact import sanitize_chart


def test_census_has_ten_fictional_charts():
    assert len(CHARTS) == 10
    mrns = {c.mrn for c in CHARTS}
    assert mrns == {f"SYN-{n}" for n in range(4401, 4411)}


def test_sanitize_replaces_name_and_mrn():
    chart = get_chart("syn-4401")
    out = sanitize_chart(chart)
    assert chart.display_name not in out
    assert chart.mrn not in out
    assert "<PATIENT>" in out
    assert "<MRN>" in out


def test_unknown_chart_raises():
    try:
        get_chart("not-a-chart")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
