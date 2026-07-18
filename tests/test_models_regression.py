"""Tests del sub-tab Regression de Models (app/tabs/models/regression/).

La garantía fuerte de esta extracción es la comparación AST (993 llamadas
idénticas al original), hecha fuera de pytest. Aquí: firma y que el despacho por
segmented_control carga sin entrenar (models=None -> ninguna rama). La cobertura
de cada rama llegará al subdividir en ols/linear/nonlinear/ensemble.
"""

import inspect
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.models.regression as regression

PREDICTORS = [
    "REM hrs",
    "Stress_prev_day",
    "Deep hrs",
    "Wake Count",
    "Sleep_hr_surplus",
    "Respiration",
    "Stress_sleep",
]


@pytest.fixture
def df_model():
    n = 60
    rng = np.random.default_rng(0)
    data = {"Date": pd.date_range("2026-01-01", periods=n, freq="D")}
    for c in PREDICTORS:
        data[c] = rng.normal(0, 1, n)
    data["Score"] = rng.normal(80, 6, n)
    data["Quality"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_render_signature():
    assert list(inspect.signature(regression.render).parameters) == ["df_model", "predictors"]


def test_render_no_model_selected_does_not_train(monkeypatch, df_model):
    m = MagicMock()
    m.segmented_control.return_value = None  # ninguna rama seleccionada
    monkeypatch.setattr(regression, "st", m)
    regression.render(df_model, PREDICTORS)  # no debe lanzar
    m.segmented_control.assert_called_once()
