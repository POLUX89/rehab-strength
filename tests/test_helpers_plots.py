"""Tests de los helpers de visualización extraídos a app/helpers/plots.py.

plot_line / plot_two_axis renderizan vía st.pyplot y correlation_insight usa
st.success/info/warning; se mockea Streamlit con monkeypatch.
"""

import matplotlib

matplotlib.use("Agg")  # backend sin display para los tests

import pandas as pd
import pytest

from app.helpers import plots


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=5),
            "y": [1, 2, 3, 4, 5],
            "z": [5, 4, 3, 2, 1],
        }
    )


def test_plot_line_calls_st_pyplot(monkeypatch, sample_df):
    calls = []
    monkeypatch.setattr(plots.st, "pyplot", lambda fig: calls.append(fig))
    plots.plot_line(sample_df, "Date", "y", "title", "ylabel")
    assert len(calls) == 1  # renderizó exactamente una figura


def test_plot_two_axis_calls_st_pyplot(monkeypatch, sample_df):
    calls = []
    monkeypatch.setattr(plots.st, "pyplot", lambda fig: calls.append(fig))
    plots.plot_two_axis(sample_df, "Date", "y", "z", "t", "y1", "y2")
    assert len(calls) == 1


def test_correlation_insight_insufficient_data():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert plots.correlation_insight(df, "a", "b") == "Insufficient data for correlation analysis."
    assert (
        plots.correlation_insight(None, "a", "b") == "Insufficient data for correlation analysis."
    )


def test_correlation_insight_strong_positive(monkeypatch):
    captured = {}
    monkeypatch.setattr(plots.st, "success", lambda msg: captured.setdefault("success", msg))
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})  # corr = 1.0
    plots.correlation_insight(df, "a", "b")
    assert "Perfect positive correlation" in captured["success"]


def test_correlation_insight_strong_negative(monkeypatch):
    captured = {}
    monkeypatch.setattr(plots.st, "success", lambda msg: captured.setdefault("success", msg))
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})  # corr = -1.0
    plots.correlation_insight(df, "a", "b")
    assert "Perfect negative correlation" in captured["success"]
