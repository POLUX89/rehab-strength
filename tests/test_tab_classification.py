"""Tests del despachador Classification (app/tabs/models/classification/).

Cubre el contrato del despachador: construye el target binario (Score < 80),
arma el split temporal (últimas H filas como test, sin barajar) y delega en la
sub-rama elegida. logit.render se mockea: el entrenamiento real (GridSearchCV)
queda fuera del alcance de estos tests.
"""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import app.tabs.models.classification as classification_tab

PREDICTORS = ["REM hrs", "Deep hrs"]


def _df_model(n):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "REM hrs": rng.normal(1.5, 0.2, n),
            "Deep hrs": rng.normal(1.2, 0.2, n),
            # Alterna clases: 1 de cada 3 filas queda bajo 80 (Bad Sleep).
            "Score": np.where(np.arange(n) % 3 == 0, 70.0, 85.0),
        }
    )


def _mock_st(monkeypatch, branch="Logistic Regression"):
    m = MagicMock()
    m.segmented_control.return_value = branch
    monkeypatch.setattr(classification_tab, "st", m)
    return m


def test_render_builds_binary_target_and_temporal_split(monkeypatch):
    _mock_st(monkeypatch)
    fake_logit = MagicMock()
    monkeypatch.setattr(classification_tab.logit, "render", fake_logit)

    df = _df_model(60)
    classification_tab.render(df, PREDICTORS)

    # Target binario: 1 si Score < 80, 0 si no.
    assert "Score_binary" in df.columns
    expected = (df["Score"] < 80).astype(int)
    assert df["Score_binary"].equals(expected)
    assert set(df["Score_binary"].unique()) == {0, 1}

    # Delegó en logit con el split empaquetado.
    fake_logit.assert_called_once()
    split = fake_logit.call_args.args[0]
    assert set(split) == {"X_train", "X_test", "y_train", "y_test"}

    # Split temporal: test = últimas 40 filas, train = las 20 primeras,
    # sin barajar (los índices conservan el orden original).
    assert len(split["X_test"]) == 40
    assert len(split["X_train"]) == 20
    assert split["y_test"].equals(df["Score_binary"].iloc[-40:])
    assert split["y_train"].equals(df["Score_binary"].iloc[:20])
    assert list(split["X_train"].columns) == PREDICTORS


def test_render_stops_when_not_enough_samples(monkeypatch):
    m = _mock_st(monkeypatch)
    # st.stop real corta el script lanzando una excepción; el mock la simula.
    m.stop.side_effect = RuntimeError("st.stop")
    fake_logit = MagicMock()
    monkeypatch.setattr(classification_tab.logit, "render", fake_logit)

    with pytest.raises(RuntimeError, match="st.stop"):
        classification_tab.render(_df_model(30), PREDICTORS)  # < 40 test + 10 train

    assert m.warning.called
    fake_logit.assert_not_called()


def test_render_delegates_to_nonlinear(monkeypatch):
    # La rama Non Linear ya no es stub: se mockea su render (el entrenamiento
    # real con GridSearchCV queda fuera del alcance de estos tests).
    _mock_st(monkeypatch, branch="Non Linear Models")
    fake_logit = MagicMock()
    fake_nonlinear = MagicMock()
    monkeypatch.setattr(classification_tab.logit, "render", fake_logit)
    monkeypatch.setattr(classification_tab.nonlinear_classification, "render", fake_nonlinear)

    classification_tab.render(_df_model(60), PREDICTORS)

    fake_nonlinear.assert_called_once()
    assert set(fake_nonlinear.call_args.args[0]) == {"X_train", "X_test", "y_train", "y_test"}
    fake_logit.assert_not_called()


def test_render_stub_branch_shows_placeholder(monkeypatch):
    # Bagging & Boosting sigue siendo stub: muestra el placeholder "coming soon".
    m = _mock_st(monkeypatch, branch="Bagging & Boosting Models")
    fake_logit = MagicMock()
    fake_nonlinear = MagicMock()
    monkeypatch.setattr(classification_tab.logit, "render", fake_logit)
    monkeypatch.setattr(classification_tab.nonlinear_classification, "render", fake_nonlinear)

    classification_tab.render(_df_model(60), PREDICTORS)

    assert m.info.called  # placeholder "coming soon"
    fake_logit.assert_not_called()
    fake_nonlinear.assert_not_called()
