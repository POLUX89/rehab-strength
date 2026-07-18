"""Tests de la tab Stats extraída a app/tabs/stats.py.

Stats tiene muchos widgets condicionales. Se mockea Streamlit con valores
sensatos (selectbox -> "Score", botones -> False para no disparar los tests de
hipótesis interactivos) y matplotlib real (Agg), para verificar que el cuerpo
descriptivo corre end-to-end sin excepción.
"""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.stats as stats_tab

ALLOWED = [
    "InBed hrs",
    "Asleep hrs",
    "Wake Count",
    "REM hrs",
    "Light hrs",
    "Deep hrs",
    "Fall Asleep",
    "Overnight HRV",
    "Stress",
    "RHR",
    "Score",
]


@pytest.fixture
def recovery_df():
    n = 120
    rng = np.random.default_rng(0)
    data = {"Date": pd.date_range("2026-01-01", periods=n, freq="D")}
    for c in ALLOWED:
        data[c] = rng.normal(50, 10, n)
    data["Score"] = rng.normal(80, 6, n)
    return pd.DataFrame(data)


@pytest.fixture
def workouts_df():
    n = 120
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "DURATION_MIN": rng.integers(30, 90, n),
            "WEIGHT_LBS": rng.integers(50, 120, n),
            "VOLUME": rng.integers(500, 2000, n),
            "RPE": rng.integers(6, 10, n),
            "est_1RM": rng.normal(120, 10, n),
        }
    )


def make_st(monkeypatch):
    m = MagicMock()
    m.selectbox.return_value = "Score"
    m.button.return_value = False  # no dispara los tests de hipótesis interactivos
    m.checkbox.return_value = False
    m.slider.return_value = 20
    m.segmented_control.return_value = True

    # st.columns(n) o st.columns([ratios]) -> n context managers
    def columns(spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(MagicMock() for _ in range(n))

    m.columns.side_effect = columns
    monkeypatch.setattr(stats_tab, "st", m)
    return m


def test_render_runs_without_error(monkeypatch, recovery_df, workouts_df):
    make_st(monkeypatch)
    stats_tab.render(recovery_df, workouts_df)  # no debe lanzar


def test_render_reads_selected_metric(monkeypatch, recovery_df, workouts_df):
    m = make_st(monkeypatch)
    stats_tab.render(recovery_df, workouts_df)
    # el selectbox de métrica se consultó al menos una vez
    assert m.selectbox.called
