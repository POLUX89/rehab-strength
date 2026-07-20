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


def _agnostic_predict_fn(model):
    """Pick the continuous output to explain for the model-agnostic route.

    Prefers ``P(positive class)`` when ``predict_proba`` is usable, then the
    signed decision margin (SVC with ``probability=False`` has no proba but does
    expose ``decision_function``), and finally the raw prediction (regression).
    """
    final = model[-1]
    if hasattr(final, "predict_proba") and getattr(final, "probability", True):
        return lambda X: model.predict_proba(X)[:, 1]
    if hasattr(final, "decision_function"):
        return model.decision_function
    return model.predict


@st.cache_data(show_spinner="Computing SHAP values… (cached per dataset)")
def compute_shap_values(_model, X_background, X_explain, cache_key, _predict_fn=None):
    """Compute SHAP values for a fitted pipeline's champion, cached per dataset.

    Routes by the final estimator's type: tree-based additive models use
    ``shap.TreeExplainer`` (fast, exact); everything else (linear models, KNN,
    SVR/SVM, AdaBoost) uses the model-agnostic ``shap.Explainer`` on the
    prediction function, which yields attributions in the original predictors.

    ``_model`` and ``_predict_fn`` are passed unhashed (leading underscore), so
    the cache is keyed ONLY on ``X_background``, ``X_explain`` and ``cache_key``.
    This is one shared cached function, so ``cache_key`` is REQUIRED and MUST
    uniquely identify the model: every sub-branch reuses the same split (same
    background/test), so without a distinct key they collide and one sub-branch
    would show another's Explanation. Use a stable per-sub-branch id, plus any
    UI toggle that changes the model with the same data (e.g. SMOTE): e.g.
    ``"reg_linear"`` or ``("cla_logit", synthetic)``.

    Assumption: tree pipelines do NOT transform features (no scaler), so the
    tree sees the original predictors and TreeExplainer can run on the raw X.
    This holds across all current sub-branches. If a tree pipeline ever gains a
    scaler, transform the data first (``_model[:-1].transform(...)``) and
    re-attach ``feature_names``.

    Args:
        _model: Fitted pipeline whose last step is the estimator (not hashed).
        X_background: Background sample for the explainer.
        X_explain: Rows to explain.
        cache_key: REQUIRED hashable that uniquely identifies the model (per
            sub-branch, plus any toggle that changes it). Not used in the body —
            only to key the shared cache and avoid cross-sub-branch collisions.
        _predict_fn: Optional prediction function for the model-agnostic route.
            Defaults to :func:`_agnostic_predict_fn` — ``P(positive class)`` for
            classifiers, the decision margin for SVC without proba, or ``predict``
            for regressors. Pass a custom callable to override (not hashed).

    Returns:
        A SHAP ``Explanation`` for ``X_explain``.
    """
    final_estimator = _model[-1]  # last pipeline step
    if isinstance(final_estimator, _TREE_MODELS):
        shap_values = shap.TreeExplainer(final_estimator, X_background)(X_explain)
        if shap_values.values.ndim == 3:
            # Tree classifier → (n, features, n_classes); keep the positive class
            # so plots match the 2-D shape of the model-agnostic route.
            shap_values = shap_values[..., 1]
        return shap_values
    predict_fn = _predict_fn if _predict_fn is not None else _agnostic_predict_fn(_model)
    return shap.Explainer(predict_fn, X_background)(X_explain)
