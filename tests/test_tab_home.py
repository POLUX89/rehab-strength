"""Tests de la tab Home (Weekly Snapshot) extraída a app/tabs/home.py.

matplotlib real (Agg) + mock de Streamlit. Verifica firma, que el cuerpo corre
end-to-end con datos válidos, y el contrato de devolver sleep como copia ordenada.
"""

import inspect
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.home as home


def make_st(monkeypatch):
    m = MagicMock()
    m.segmented_control.return_value = 14
    m.slider.return_value = 5
    m.columns.side_effect = lambda spec, *a, **k: tuple(
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    )
    monkeypatch.setattr(home, "st", m)
    return m


@pytest.fixture
def frames():
    n = 40
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    workouts = pd.DataFrame(
        {
            "Date": dates,
            "DATE": dates,
            "DAY": dates,
            "WORKOUT_NAME": ["Push"] * n,
            "DURATION_MIN": rng.integers(30, 90, n),
        }
    )
    recovery = pd.DataFrame(
        {
            "Date": dates,
            "Sigmoid Recovery Score": rng.uniform(0.4, 0.9, n),
            "Sigmoid with Nap": rng.uniform(0.4, 0.9, n),
            "DELTA_NAP": rng.normal(0, 0.05, n),
            "Nap_Status": ["Good"] * n,
            "Score": rng.normal(80, 5, n),
            "Overnight HRV": rng.normal(55, 8, n),
        }
    )
    sleep = pd.DataFrame({"Date": dates, "Asleep_Nap": rng.integers(0, 40, n)})
    return workouts, sleep, recovery


def test_render_signature():
    assert list(inspect.signature(home.render).parameters) == ["workouts", "sleep", "recovery"]


def test_render_runs_and_returns_sorted_sleep(monkeypatch, frames):
    make_st(monkeypatch)
    workouts, sleep, recovery = frames
    shuffled = sleep.sample(frac=1, random_state=2)
    out = home.render(workouts, shuffled, recovery)
    assert out is not shuffled  # copia
    assert list(out["Date"]) == sorted(out["Date"])  # ordenada por Date
