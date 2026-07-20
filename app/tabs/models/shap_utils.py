"""Helper de SHAP compartido por las sub-ramas de regression y classification.

Un único punto para calcular SHAP del campeón de cada sub-rama, evitando las 3
(pronto 6) copias del mismo patrón. Vive un nivel arriba de regression/ y
classification/ para que ambos lo importen sin ciclos (los submódulos importan
de aquí; este módulo no importa submódulos).
"""

from __future__ import annotations

import shap
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Modelos aditivos basados en árboles → shap.TreeExplainer (rápido y exacto).
# NO incluir AdaBoost: su predicción por mediana ponderada no es aditiva y
# TreeExplainer la rechaza (InvalidModelError) → cae a la ruta model-agnostic.
_TREE_MODELS = (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)


@st.cache_data(show_spinner="Computing SHAP values… (cached per dataset)")
def compute_shap_values(_model, X_background, X_explain, _predict_fn=None, cache_key=None):
    """Compute SHAP values for a fitted pipeline's champion, cached per dataset.

    Routes by the final estimator's type: tree-based additive models use
    ``shap.TreeExplainer`` (fast, exact); everything else (linear models, KNN,
    SVR/SVM, AdaBoost) uses the model-agnostic ``shap.Explainer`` on the
    prediction function, which yields attributions in the original predictors.

    ``_model`` and ``_predict_fn`` are passed unhashed (leading underscore):
    the cache is keyed on ``X_background``, ``X_explain`` and ``cache_key``.
    That suffices when the model is determined by the data alone — but when it
    also depends on something else with the SAME data (e.g. a SMOTE toggle in
    classification), pass that value as ``cache_key`` so toggling actually
    recomputes instead of returning a stale Explanation.

    Assumption: tree pipelines do NOT transform features (no scaler), so the
    tree sees the original predictors and TreeExplainer can run on the raw X.
    This holds across all current sub-branches. If a tree pipeline ever gains a
    scaler, transform the data first (``_model[:-1].transform(...)``) and
    re-attach ``feature_names``.

    Args:
        _model: Fitted pipeline whose last step is the estimator (not hashed).
        X_background: Background sample for the explainer.
        X_explain: Rows to explain.
        _predict_fn: Optional prediction function for the model-agnostic route
            (e.g. ``model.predict_proba`` in classification). Defaults to
            ``_model.predict`` (not hashed).
        cache_key: Optional hashable disambiguating the cache when the model
            depends on more than the data (e.g. a UI toggle). Not used in the
            body — only to key the cache.

    Returns:
        A SHAP ``Explanation`` for ``X_explain``.
    """
    final_estimator = _model[-1]  # last pipeline step
    if isinstance(final_estimator, _TREE_MODELS):
        return shap.TreeExplainer(final_estimator, X_background)(X_explain)
    predict_fn = _predict_fn if _predict_fn is not None else _model.predict
    return shap.Explainer(predict_fn, X_background)(X_explain)
