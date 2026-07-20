"""Sub-rama 'Non Linear' de Classification (Decision Tree, KNN, SVM).

render(split) entrena y tunea Decision Tree, KNN y SVM con GridSearchCV sobre
TimeSeriesSplit (CV temporal). KNN y SVM se escalan (StandardScaler); el árbol
no lo necesita. SMOTE opcional se aplica dentro del pipeline (solo al train de
cada fold, sin fuga). scoring F2. El tuning se cachea por (datos, toggle SMOTE).
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.tabs.models.shap_utils import compute_shap_values


def _synthetic():
    """Render the SMOTE toggle checkbox.

    Returns:
        True if the user enabled SMOTE oversampling, else False.
    """
    return st.checkbox(
        "Balance classes with SMOTE (experimental)",
        False,
        key="nonlinear_smote_toggle",
    )


def _build_pipe(name, estimator, scale, synthetic):
    """Assemble an imbalanced-learn pipeline: [scaler] → [SMOTE] → estimator.

    The scaler goes before SMOTE (SMOTE relies on k-NN distances), and SMOTE is
    applied only to the training portion of each CV fold, so no synthetic rows
    leak into validation or test.

    Args:
        name: Step name for the estimator (also the grid prefix, e.g. ``"svm"``).
        estimator: The sklearn classifier to tune.
        scale: Whether to prepend a StandardScaler (True for KNN/SVM).
        synthetic: Whether to insert a SMOTE step.

    Returns:
        An ``imblearn`` Pipeline ready for GridSearchCV.
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if synthetic:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
    steps.append((name, estimator))
    return ImbPipeline(steps)


def _row(name, best_estimator, best_params, y_train, y_pred_train, y_test, y_pred_test):
    """Flatten one model's train/test metrics into a single row dict.

    Args:
        name: Display name used as the row index.
        best_estimator: The fitted best estimator from GridSearchCV.
        best_params: The model's best hyperparameters.
        y_train: True training labels.
        y_pred_train: Predicted training labels.
        y_test: True test labels.
        y_pred_test: Predicted test labels.

    Returns:
        A dict with accuracy/precision/recall/F1/F2 for train and test, plus
        ``best_params`` and ``best_estimator``.
    """
    return {
        "model": name,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_precision": precision_score(y_train, y_pred_train, zero_division=0),
        "train_recall": recall_score(y_train, y_pred_train, zero_division=0),
        "train_f1": f1_score(y_train, y_pred_train, zero_division=0),
        "train_f2": fbeta_score(y_train, y_pred_train, beta=2, zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
        "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
        "test_f1": f1_score(y_test, y_pred_test, zero_division=0),
        "test_f2": fbeta_score(y_test, y_pred_test, beta=2, zero_division=0),
        "best_params": best_params,
        "best_estimator": best_estimator,
    }


@st.cache_data(show_spinner="Fitting non-linear models... (runs once per dataset)")
def _fit_nonlinear(split, synthetic=False):
    """Fit and tune Decision Tree, KNN and SVM with time-aware cross-validation.

    Builds one pipeline per model ([scaler] → [SMOTE] → estimator), grid-searches
    each over F2 with ``TimeSeriesSplit``, and collects train/test metrics.
    Cached on (data, SMOTE toggle), so it only refits when those change.

    Args:
        split: Dict with ``X_train``/``y_train``/``X_test``/``y_test``.
        synthetic: If True, insert SMOTE into every pipeline.

    Returns:
        A ``(results_df, elapsed_seconds, winner_name)`` tuple; ``results_df``
        is indexed by model name and ``winner_name`` is the model with the
        highest test F2 (its fitted estimator lives in
        ``results_df["best_estimator"]``).
    """
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    time0 = time.time()
    f2 = make_scorer(fbeta_score, beta=2)
    tscv = TimeSeriesSplit(n_splits=3)
    rows = []

    # Imbalance handling, mirroring logit: with SMOTE off, weight classes
    # inversely to frequency; with SMOTE on, leave it None to avoid
    # double-correcting. KNN has no class_weight, so it can only be balanced
    # via SMOTE (surfaced to the user in render()).
    cw = None if synthetic else "balanced"

    # ---- Decision Tree (trees don't need scaling) ----
    full_tree = DecisionTreeClassifier(random_state=42, class_weight=cw)
    ccp_alphas = full_tree.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
    samples_ccp = np.quantile(ccp_alphas, np.linspace(0, 1, 10))
    grid_dt = GridSearchCV(
        _build_pipe(
            "dt",
            DecisionTreeClassifier(random_state=42, class_weight=cw),
            scale=False,
            synthetic=synthetic,
        ),
        param_grid={
            "dt__criterion": ["gini", "entropy"],
            "dt__max_depth": [3, 5, 7, 9, None],
            "dt__min_samples_split": [2, 5, 10],
            "dt__ccp_alpha": samples_ccp,
        },
        cv=tscv,
        scoring=f2,
    )
    grid_dt.fit(X_train, y_train)
    rows.append(
        _row(
            "Decision Tree",
            grid_dt.best_estimator_,
            grid_dt.best_params_,
            y_train,
            grid_dt.predict(X_train),
            y_test,
            grid_dt.predict(X_test),
        )
    )

    # ---- KNN (scaled) ----
    grid_knn = GridSearchCV(
        _build_pipe("knn", KNeighborsClassifier(), scale=True, synthetic=synthetic),
        param_grid={
            "knn__n_neighbors": [3, 5, 7, 9],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2],
        },
        cv=tscv,
        scoring=f2,
    )
    grid_knn.fit(X_train, y_train)
    rows.append(
        _row(
            "KNN",
            grid_knn.best_estimator_,
            grid_knn.best_params_,
            y_train,
            grid_knn.predict(X_train),
            y_test,
            grid_knn.predict(X_test),
        )
    )

    # ---- SVM (scaled) ----
    grid_svm = GridSearchCV(
        _build_pipe("svm", SVC(random_state=42, class_weight=cw), scale=True, synthetic=synthetic),
        param_grid=[
            {  # linear — no gamma, no degree
                "svm__kernel": ["linear"],
                "svm__C": np.logspace(-3, 3, 30),
            },
            {  # rbf and sigmoid — gamma applies, degree does not
                "svm__kernel": ["rbf", "sigmoid"],
                "svm__C": np.logspace(-3, 3, 30),
                "svm__gamma": ["scale", "auto"],
            },
            {  # poly — both gamma and degree apply
                "svm__kernel": ["poly"],
                "svm__C": np.logspace(-3, 3, 10),
                "svm__gamma": ["scale", "auto"],
                "svm__degree": [2, 3],
            },
        ],
        cv=tscv,
        scoring=f2,
    )
    grid_svm.fit(X_train, y_train)
    rows.append(
        _row(
            "SVM",
            grid_svm.best_estimator_,
            grid_svm.best_params_,
            y_train,
            grid_svm.predict(X_train),
            y_test,
            grid_svm.predict(X_test),
        )
    )

    winner = max(rows, key=lambda r: r["test_f2"])["model"]  # name; estimator is in results_df
    results_df = pd.DataFrame(rows).set_index("model")
    return results_df, time.time() - time0, winner


def graph_winner(results, title="Non-Linear Models"):
    """Graph the winner among the compared models.

    Args:
        results: DataFrame with train/test metrics for each model.
        title: Model-family label used in the chart's suptitle. Defaults to
            ``"Non-Linear Models"``; the ensemble branch passes its own.

    Returns:
        None.
    """
    winner = results["test_f2"].idxmax()
    st.subheader(f"🏆 Winner: {winner} (highest F2 on test set)", divider=True)
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    sns.barplot(
        data=results.reset_index(),
        x="model",
        y="train_f2",
        palette="icefire",
        ax=ax[0],
        legend=False,
        hue="model",
    )
    bars = ax[0].patches
    for bar in bars:
        height = bar.get_height()
        ax[0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    sns.barplot(
        data=results.reset_index(), x="model", y="test_f2", palette="icefire", ax=ax[1], hue="model"
    )
    bars = ax[1].patches
    for bar in bars:
        height = bar.get_height()
        ax[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax[0].xaxis.set_ticklabels(results.index, fontsize=6, rotation=45)
    ax[0].set_title("Train F2 Score", fontsize=8, fontweight="bold")
    ax[0].set_xlabel("")
    ax[0].set_ylabel("F2 Score", fontsize=10, fontweight="bold")
    ax[0].set_ylim(0, 1)
    ax[1].set_title("Test F2 Score", fontsize=8, fontweight="bold")
    ax[1].set_xlabel("")
    ax[1].xaxis.set_ticklabels(results.index, fontsize=6, rotation=45)
    ax[1].set_ylabel("")
    ax[1].set_ylim(0, 1)
    plt.suptitle(f"Train vs Test F2 Scores for {title}", fontsize=14, fontweight="bold", y=1.05)
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)


def metrics_table(results):
    """Render each model's train vs test metrics as a 5×2 st.metric grid.

    One expander per model: the first row of five columns holds the train
    metrics, the second row the test metrics with the test−train delta so
    over/underfitting shows at a glance.

    Args:
        results: DataFrame indexed by model, with ``train_*``/``test_*``
            columns for accuracy, precision, recall, f1 and f2.

    Returns:
        None.
    """
    with st.expander("📈 Train vs Test Metrics for Each Model", expanded=True):
        keys = ["accuracy", "precision", "recall", "f1", "f2"]
        labels = ["Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"]
        for model_name, row in results.iterrows():
            with st.expander(f"📈 {model_name} — Train vs Test", expanded=True):
                st.markdown("**Train**")
                for col, key, label in zip(st.columns(5), keys, labels, strict=True):
                    col.metric(label, f"{row[f'train_{key}']:.3f}")
                st.markdown("**Test**")
                for col, key, label in zip(st.columns(5), keys, labels, strict=True):
                    train_v, test_v = row[f"train_{key}"], row[f"test_{key}"]
                    col.metric(label, f"{test_v:.3f}", delta=f"{test_v - train_v:+.3f}")


def render(split):
    """Render the Non Linear Models sub-branch.

    Reads the SMOTE toggle, fits Decision Tree, KNN and SVM (scaling for the
    latter two, optional SMOTE), and shows a train-vs-test metrics table.

    Args:
        split: Dict with ``X_train``/``y_train``/``X_test``/``y_test``.

    Returns:
        None.
    """
    st.title("Non Linear Models")
    synthetic = _synthetic()  # widget read OUTSIDE the cached function
    if not synthetic:
        st.caption(
            "Imbalance handled with `class_weight='balanced'` (Decision Tree, SVM). "
            "KNN has no `class_weight` — enable SMOTE to balance it."
        )
    results, elapsed, winner = _fit_nonlinear(split, synthetic)
    results = results.sort_values(by="test_f2", ascending=False)
    st.caption(f"Fitted in {elapsed:.2f}s (cached unless data or the SMOTE toggle change).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Performance", divider=True)
        st.dataframe(results.drop(columns="best_estimator"))
    with c2:
        st.subheader("Model Selection", divider=True)
        st.write(f"🏆 Winner: {winner} — highest F2 on test set")
        best_estimator = results.loc[winner, "best_estimator"]
        cm = confusion_matrix(split["y_test"], best_estimator.predict(split["X_test"]))
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Good", "Bad"],
            yticklabels=["Good", "Bad"],
        )
        ax.set(title=f"Confusion Matrix — {winner} (test)", xlabel="Predicted", ylabel="Actual")
        st.pyplot(fig)
        plt.close(fig)
    graph_winner(results)
    metrics_table(results)

    with st.expander("📊 Explanatory Power of Predictors", expanded=True):
        st.subheader(f"📊 Explanatory Power of Predictors — {winner}")
        st.caption(
            "SHAP for the winning model. TreeExplainer if a Decision Tree wins "
            "(fast, exact); otherwise model-agnostic on the probability / decision "
            "margin of **Bad Sleep**."
        )
        # The shared helper routes by model type and picks the output to explain
        # (P(Bad Sleep), or the SVM decision margin). cache_key=synthetic so
        # toggling SMOTE recomputes instead of returning a stale Explanation.
        X_background = shap.sample(split["X_train"], 100, random_state=42)
        shap_values = compute_shap_values(
            best_estimator, X_background, split["X_test"], cache_key=("cla_nonlinear", synthetic)
        )
        sample_ind = -1  # last sample in the test set

        force_plot = shap.plots.force(shap_values[sample_ind], matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for last sample {split['X_test'].index[sample_ind]}")
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
            plt.title(f"SHAP Waterfall Plot for last sample {split['X_test'].index[sample_ind]}")
            st.pyplot(fig)
            plt.close(fig)
