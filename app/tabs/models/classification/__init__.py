"""Classification — despachador del tipo de modelo.

render(df_model, predictors) muestra el segmented_control de tipo de modelo y
delega en las sub-ramas: logit (Logistic Regression), nonlinear (Non Linear) y
ensemble (Bagging & Boosting). Aquí se construye el target binario y el split
temporal (train/test) una sola vez; cada sub-rama recibe ese `split`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split

from . import logit, nonlinear_classification


# knn, dt, svm
# rf, adaboost
def render(df_model, predictors):
    """Render the Classification dispatcher.

    Builds the binary target (``Score < 80`` → 1 = "Bad Sleep") and the
    temporal train/test split once, plots the class distributions, then
    delegates to the chosen model sub-branch (logit, nonlinear, ensemble).

    Args:
        df_model: Model-ready DataFrame including ``Score`` and predictors.
        predictors: Predictor column names.

    Returns:
        None.
    """
    # Design target variable for classification: Score < 80 (1) vs Score >= 80 (0)
    df_model["Score_binary"] = (df_model["Score"] < 80).astype(
        int
    )  # Binary target: 1 if Score < 80, else 0
    H = 40  # temporal test window (rows)
    MIN_TRAIN = 10
    if len(df_model) < H + MIN_TRAIN:
        st.warning(
            f"Not enough samples for classification: {len(df_model)} rows available, "
            f"but the temporal split needs at least {H + MIN_TRAIN} "
            f"({H} test + {MIN_TRAIN} train). Load more data.",
            icon="⚠️",
        )
        st.stop()
    X = df_model[predictors]
    y = df_model["Score_binary"]  # Binary target: 1 if Score < 80, else 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=H, shuffle=False, stratify=None
    )
    split = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Plot the entire dataset distribution of the binary target variable
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_model, x="Score_binary", palette="icefire", hue="Score_binary")
    bars = plt.gca().patches
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    plt.xticks(ticks=[0, 1], labels=["Good Sleep (0)", "Bad Sleep (1)"], fontsize=10)
    plt.ylabel("Count", fontsize=12, fontweight="bold")
    plt.xlabel("")
    plt.title("Score Quality Distribution", fontsize=14, fontweight="bold", pad=20)
    sns.despine()
    plt.legend(title="Score Quality", labels=["Good Sleep (0)", "Bad Sleep (1)"], loc="upper right")
    st.pyplot(fig)
    plt.close(fig)

    # Plot the train and test distribution of the binary target variable
    st.subheader("Train and Test Distribution of the Binary Target Variable:", divider=True)
    fig, ax = plt.subplots(1, 2, figsize=(4, 3))
    plt.suptitle(
        "Train and Test Distribution of the Binary Target Variable",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    sns.countplot(x=y_train, ax=ax[0], palette="icefire", hue=y_train, legend=False)
    bars = ax[0].patches
    for bar in bars:
        height = bar.get_height()
        ax[0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax[0].set_title("Training Set Distribution", fontsize=6, fontweight="bold")
    ax[0].set_xlabel("")
    ax[0].xaxis.set_ticklabels(["Good Sleep (0)", "Bad Sleep (1)"], fontsize=6)
    sns.countplot(x=y_test, ax=ax[1], palette="icefire", hue=y_test, legend=True)
    bars = ax[1].patches
    for bar in bars:
        height = bar.get_height()
        ax[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax[1].legend(labels=["Good Sleep (0)", "Bad Sleep (1)"], loc="upper right", fontsize=4)
    ax[1].set_title("Test Set Distribution", fontsize=6, fontweight="bold")
    ax[1].set_ylabel("")
    ax[1].set_xlabel("")
    ax[1].xaxis.set_ticklabels(["Good Sleep (0)", "Bad Sleep (1)"], fontsize=6)
    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "Note: Class Imbalance is detected affecting model performance. Consider using techniques like SMOTE or class weighting to address this issue.",
        icon="⚠️",
    )

    models = st.segmented_control(
        "Select Model Type:",
        [
            "Logistic Regression",
            "Non Linear Models",
            "Bagging & Boosting Models",
        ],
        key="model_type_control",
        default="Logistic Regression",
    )
    if models == "Logistic Regression":
        logit.render(split)
    elif models == "Non Linear Models":
        nonlinear_classification.render(split)
    elif models == "Bagging & Boosting Models":
        st.info("🚧 Bagging & Boosting Models (Random Forest, AdaBoost) — coming soon.")
        # ensemble.render(split)
