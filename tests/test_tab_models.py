"""Tests de la tab Models extraída a app/tabs/models/ (fase 1: módulo entero).

Models es enorme y muy dependiente de estado (session_state, GridSearch, SHAP);
un test que ejecute cada rama es inviable aquí. La garantía fuerte de esta fase
es la comparación AST (multiset de llamadas idéntico al original) hecha fuera de
pytest. Aquí se cubre: contrato de import/firma y que la preparación inicial de
datos corre con session_state real y los botones apagados.
"""

import inspect
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import app.tabs.models as models_tab


def test_render_exists_and_signature():
    assert callable(models_tab.render)
    params = list(inspect.signature(models_tab.render).parameters)
    assert params == ["time_series_analysis"]


class _SessionState(dict):
    """session_state soporta acceso por atributo y por clave, como en Streamlit."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _recovery_df():
    n = 60
    rng = np.random.default_rng(0)
    cols = {
        "Date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "Sleep Need": ["7h 30min"] * n,
        "Efficiency": ["90%"] * n,
        "Asleep hrs": rng.normal(7, 0.5, n),
        "REM hrs": rng.normal(1.5, 0.2, n),
        "Deep hrs": rng.normal(1.2, 0.2, n),
        "Wake Count": rng.integers(0, 4, n),
        "Stress_prev_day": rng.normal(30, 5, n),
        "Respiration": rng.normal(14, 1, n),
        "Stress_sleep": rng.normal(25, 5, n),
        "Score": rng.normal(80, 6, n),
        "Quality": rng.integers(0, 2, n),
    }
    return pd.DataFrame(cols)


def test_render_initial_prep_runs(monkeypatch):
    # botones/selectbox apagados: no entra al entrenamiento pesado, solo la
    # preparación inicial (re-lee recovery, arma df_model, muestra el veredicto TSA).
    m = MagicMock()
    m.session_state = _SessionState(df_recovery=_recovery_df())
    # La rama Classification ya no está vacía: se mockea el submódulo para que
    # este test siga siendo solo de preparación + despacho (la lógica interna
    # se cubre en test_tab_classification.py).
    m.selectbox.return_value = "Classification"
    m.button.return_value = False
    m.checkbox.return_value = False
    m.segmented_control.return_value = "OLS diagnosis"
    m.columns.side_effect = lambda spec, *a, **k: tuple(
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    )
    monkeypatch.setattr(models_tab, "st", m)
    fake_classification = MagicMock()
    monkeypatch.setattr(models_tab, "classification", fake_classification)
    models_tab.render("Stationary")  # no debe lanzar
    # el veredicto TSA recibido se muestra y la rama delega en classification
    assert m.success.called
    fake_classification.render.assert_called_once()
