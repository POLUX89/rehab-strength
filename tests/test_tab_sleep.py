"""Tests de la tab Sleep extraída a app/tabs/sleep.py.

Igual que Workouts: matplotlib real (Agg) para ejecutar el cuerpo de gráficos,
mockeando solo Streamlit y plot_line.
"""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import app.tabs.sleep as sl


def make_st_mock(monkeypatch):
    m = MagicMock()
    m.columns.return_value = (MagicMock(), MagicMock())  # left, right
    m.slider.return_value = 365
    monkeypatch.setattr(sl, "st", m)
    monkeypatch.setattr(sl, "plot_line", lambda *a, **k: None)
    return m


@pytest.fixture
def sleep_df():
    dates = pd.date_range("2026-01-01", periods=8, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "Score": [80, 82, 78, 85, 90, 88, 84, 86],
            "REM hrs": [1.5, 1.6, 1.4, 1.7, 1.8, 1.6, 1.5, 1.7],
            "Light hrs": [4.0, 4.1, 3.9, 4.2, 4.3, 4.0, 4.1, 4.2],
            "Deep hrs": [1.2, 1.3, 1.1, 1.4, 1.5, 1.3, 1.2, 1.4],
            "Wake Count": [2, 3, 1, 2, 1, 2, 3, 1],
            "Asleep_Nap": [20, 25, 30, 15, 40, 35, 20, 25],
        }
    )


def test_render_none_infos_and_returns_none(monkeypatch):
    m = make_st_mock(monkeypatch)
    assert sl.render(None) is None
    m.info.assert_called_once()


def test_render_without_date_returns_same_and_errors(monkeypatch):
    m = make_st_mock(monkeypatch)
    df = pd.DataFrame({"Score": [1, 2]})
    out = sl.render(df)
    assert out is df
    m.error.assert_called_once()


def test_render_full_runs_and_returns_sorted_copy(monkeypatch, sleep_df):
    make_st_mock(monkeypatch)
    shuffled = sleep_df.sample(frac=1, random_state=1)  # desordenar
    out = sl.render(shuffled)
    assert out is not shuffled  # copia
    assert list(out["Date"]) == sorted(out["Date"])  # ordenada por Date
