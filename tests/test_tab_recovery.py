"""Tests de la tab Recovery extraída a app/tabs/recovery.py.

Se mockea Streamlit (MagicMock) y plot_line. El foco es el contrato de render():
guardas de entrada y, sobre todo, que con datos válidos devuelve una COPIA
ordenada por Date (lo que en el monolito protegía st.session_state de mutaciones
de las tabs siguientes).
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

import app.tabs.recovery as rec


@pytest.fixture
def st_mock(monkeypatch):
    m = MagicMock()
    m.multiselect.return_value = []
    monkeypatch.setattr(rec, "st", m)
    monkeypatch.setattr(rec, "plot_line", lambda *a, **k: None)
    return m


def test_render_none_returns_none_and_infos(st_mock):
    assert rec.render(None) is None
    st_mock.info.assert_called_once()


def test_render_without_date_returns_same_and_errors(st_mock):
    df = pd.DataFrame({"Score": [1, 2, 3]})
    out = rec.render(df)
    assert out is df  # sin Date no se copia ni ordena
    st_mock.error.assert_called_once()


def test_render_sorts_by_date(st_mock):
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            "Score": [3, 1, 2],
        }
    )
    out = rec.render(df)
    assert list(out["Score"]) == [1, 2, 3]  # ordenado por Date ascendente


def test_render_returns_copy_not_original(st_mock):
    # el contrato clave: render devuelve una copia, no el objeto de entrada,
    # para que las mutaciones in-place de tabs posteriores no toquen session_state.
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "Score": [1, 2]})
    out = rec.render(df)
    assert out is not df


def test_render_plots_chosen_components(monkeypatch):
    m = MagicMock()
    m.multiselect.return_value = ["Score", "Overnight HRV"]
    monkeypatch.setattr(rec, "st", m)
    calls = []
    monkeypatch.setattr(rec, "plot_line", lambda *a, **k: calls.append(a))
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Score": [1, 2],
            "Overnight HRV": [50, 55],
            "Sigmoid Recovery Score": [0.5, 0.6],
        }
    )
    rec.render(df)
    # un plot para el sigmoid + uno por cada componente elegido
    assert len(calls) == 3
