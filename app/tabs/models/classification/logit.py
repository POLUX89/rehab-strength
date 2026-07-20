"""Sub-rama 'Logit diagnosis' de Classification (Logistic Regression).

render(split) entrena y tunea una regresión logística con GridSearchCV sobre
TimeSeriesSplit (CV temporal). Pipeline StandardScaler → [SMOTE opcional] →
LogisticRegression(saga); regularización vía l1_ratio, scoring F2. El tuning se
cachea por (datos, toggle SMOTE); render muestra métricas train vs test.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.tabs.models.shap_utils import compute_shap_values


def _synthetic():
    """Render the SMOTE toggle checkbox.

    Returns:
        True if the user enabled SMOTE oversampling, else False.
    """
    return st.checkbox(
        "Balance classes with SMOTE (experimental)",
        False,
        key="logit_smote_toggle",
    )


@st.cache_data(show_spinner="Tuning logit… (runs once per dataset)")
def _tune_logit(split, synthetic=False):
    """Tune a logistic regression with time-aware cross-validation.

    Builds a ``StandardScaler → [SMOTE] → LogisticRegression(saga)`` pipeline
    and grid-searches over C, class weight and ``l1_ratio`` using
    ``TimeSeriesSplit`` and the F2 score. Cached on its arguments, so it only
    retrains when the data or the SMOTE toggle change.

    Args:
        split: Dict with ``X_train``/``y_train``/``X_test``/``y_test``.
        synthetic: If True, add SMOTE to the pipeline and drop ``class_weight``
            balancing to avoid double-correcting the imbalance.

    Returns:
        A ``(best_estimator, best_params, best_cv_score, elapsed_seconds)``
        tuple, where ``best_cv_score`` is the best F2 from cross-validation.
    """
    f2 = make_scorer(fbeta_score, beta=2)
    time0 = time.time()
    X_train, y_train = split["X_train"], split["y_train"]

    steps = [("scaler", StandardScaler())]
    if synthetic:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
        class_weights = [None]  # SMOTE ya equilibra: no re-pesar (evita doble corrección)
    else:
        class_weights = [None, "balanced"]  # sin SMOTE: dejar que la CV pruebe re-pesar
    steps.append(("logit", LogisticRegression(max_iter=5000, solver="saga")))
    estimator = ImbPipeline(steps)

    param_grid = {
        "logit__C": [0.01, 0.1, 1, 10, 100],
        "logit__class_weight": class_weights,
        "logit__l1_ratio": [0.0, 0.5, 1.0],
    }
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring=f2,
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - time0
    return grid.best_estimator_, grid.best_params_, grid.best_score_, elapsed


def _plot_f2_train_test(f2_train, f2_test):
    """Bar chart comparing the model's F2 score on train vs test.

    A large train-over-test gap flags overfitting at a glance.

    Args:
        f2_train: F2 score on the training set.
        f2_test: F2 score on the test set.

    Returns:
        None. Draws the chart with ``st.pyplot``.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(
        x=["Train", "Test"],
        y=[f2_train, f2_test],
        hue=["Train", "Test"],
        palette="icefire",
        legend=False,
        ax=ax,
    )
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("F2 Score", fontweight="bold")
    ax.set_title("F2 Score: Train vs Test", fontweight="bold")
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)


def render(split):
    """Render the Logistic Regression sub-branch.

    Trains via :func:`_tune_logit` (with the SMOTE toggle), then shows the
    best hyperparameters, the classification report, the confusion matrix and
    train-vs-test performance metrics.

    Args:
        split: Dict with ``X_train``/``y_train``/``X_test``/``y_test``.

    Returns:
        None.
    """
    st.title("Logistic Regression Model")
    X_test, y_test = split["X_test"], split["y_test"]
    synthetic = _synthetic()
    model_logit, best_params, best_cv, elapsed = _tune_logit(split, synthetic)

    with st.expander("📈 Training Metrics: Train vs Test", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Best hyperparameters:", divider=True)
            st.write(best_params)
            st.caption(
                f"Trained in {elapsed:.2f}s (cached unless data or the SMOTE toggle change)."
            )
            st.write(f"F2 score (temporal CV): {best_cv:.3f}")
            y_pred = model_logit.predict(X_test)
            f2_test = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
            st.write(f"F2 score (test): {f2_test:.3f}")
            st.subheader("Classification Report test:", divider=True)
            st.text(
                classification_report(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                    target_names=["Good Sleep", "Bad Sleep"],
                    zero_division=0,
                )
            )
        with c2:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set(title="Confusion Matrix test set", xlabel="Predicted", ylabel="Actual")
            st.pyplot(fig)
            plt.close(fig)
    with st.expander("📈 Performance Metrics: Train vs Test", expanded=True):
        y_train = split["y_train"]
        y_pred_train = model_logit.predict(split["X_train"])  # una sola vez
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred)
        precision_train = precision_score(y_train, y_pred_train, zero_division=0)
        precision_test = precision_score(y_test, y_pred, zero_division=0)
        recall_train = recall_score(y_train, y_pred_train, zero_division=0)
        recall_test = recall_score(y_test, y_pred, zero_division=0)
        f1_train = f1_score(y_train, y_pred_train, zero_division=0)
        f1_test = f1_score(y_test, y_pred, zero_division=0)
        f2_train = fbeta_score(y_train, y_pred_train, beta=2, zero_division=0)

        st.subheader("F2 Score: Train vs Test", divider=True)
        _plot_f2_train_test(f2_train, f2_test)

        st.subheader("📈 Train Set Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Accuracy", f"{accuracy_train:.3f}")
        with c2:
            st.metric("Precision", f"{precision_train:.3f}")
        with c3:
            st.metric("Recall", f"{recall_train:.3f}")
        with c4:
            st.metric("F1 Score", f"{f1_train:.3f}")
        with c5:
            st.metric("F2 Score", f"{f2_train:.3f}")

        st.subheader("📈 Test Set Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(
                "Accuracy", f"{accuracy_test:.3f}", delta=f"{accuracy_test - accuracy_train:+.3f}"
            )
        with c2:
            st.metric(
                "Precision",
                f"{precision_test:.3f}",
                delta=f"{precision_test - precision_train:+.3f}",
            )
        with c3:
            st.metric("Recall", f"{recall_test:.3f}", delta=f"{recall_test - recall_train:+.3f}")
        with c4:
            st.metric("F1 Score", f"{f1_test:.3f}", delta=f"{f1_test - f1_train:+.3f}")
        with c5:
            st.metric("F2 Score", f"{f2_test:.3f}", delta=f"{f2_test - f2_train:+.3f}")

    with st.expander("📊 Explanatory Power of Predictors", expanded=True):
        st.subheader("📊 Explanatory Power of Predictors")
        st.caption(
            "SHAP over the model's predicted **probability of Bad Sleep** — how much "
            "each predictor pushes that risk up or down."
        )
        # LogisticRegression → model-agnostic route. Explain P(Bad Sleep) (the
        # positive class), not the hard 0/1 label. cache_key=synthetic so
        # toggling SMOTE recomputes instead of returning a stale Explanation.
        X_background = shap.sample(split["X_train"], 100, random_state=42)
        shap_values = compute_shap_values(
            model_logit,
            X_background,
            X_test,
            cache_key=("cla_logit", synthetic),
            _predict_fn=lambda X: model_logit.predict_proba(X)[:, 1],
        )
        sample_ind = -1  # last sample in the test set

        force_plot = shap.plots.force(shap_values[sample_ind], matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for last sample {X_test.index[sample_ind]}")
        st.pyplot(force_plot)
        plt.close(force_plot)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.bar(shap_values, max_display=14, show=False)
            plt.title("Mean Absolute SHAP Values")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[sample_ind], max_display=14, show=False)
            plt.title(f"SHAP Waterfall Plot for last sample {X_test.index[sample_ind]}")
            st.pyplot(fig)
            plt.close(fig)
