"""Sub-rama 'Other Linear Models' de Regression (Ridge/Lasso/ElasticNet).

Extraída de models/regression sin cambiar la lógica. render(df_model, predictors).
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@st.cache_data(show_spinner="Computing SHAP values… (cached per dataset)")
def _compute_shap_values(_model, X_background, X_explain):
    """Compute model-agnostic SHAP values for a fitted estimator.

    Wrapping this in ``@st.cache_data`` avoids recomputing SHAP — the expensive
    step — on every Streamlit rerun. ``_model`` is passed unhashed (leading
    underscore): it is deterministic given the same data, so hashing
    ``X_background`` and ``X_explain`` is enough to key the cache.

    Args:
        _model: Fitted estimator/pipeline exposing ``predict`` (not hashed).
        X_background: Background sample used by the explainer.
        X_explain: Rows to explain.

    Returns:
        A SHAP ``Explanation`` for ``X_explain``.
    """
    explainer = shap.Explainer(_model.predict, X_background)
    return explainer(X_explain)


def render(df_model, predictors):
    """Render the Other Linear Models sub-branch.

    Trains OLS, Ridge, Lasso and ElasticNet (with polynomial features and
    scaling), compares them, and shows metrics and a learning curve for the
    winner.

    Args:
        df_model: Model-ready DataFrame of predictors and target.
        predictors: Predictor column names.

    Returns:
        None.
    """

    @st.cache_data(show_spinner="Fitting linear models... (runs once per dataset)")
    def fit_reg_linear_models(X_train, y_train, X_test, y_test):
        """Fit OLS, Ridge, Lasso and ElasticNet with polynomial features and scaling.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.

        Returns:
            A dict mapping each model name to its fitted model and
            performance metrics.
        """
        results = []
        # ------------------OLS with Polynomial Features------------------
        pipe_ols = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2)),
                ("ols", LinearRegression()),
            ]
        )
        best_ols = pipe_ols.fit(X_train, y_train)
        y_pred_train_ols = best_ols.predict(X_train)
        y_pred_test_ols = best_ols.predict(X_test)
        # ------------------ Ridge
        pipe_ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2)),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        grid_ridge = GridSearchCV(
            pipe_ridge,
            param_grid={"ridge__alpha": np.logspace(-3, 3, 100)},
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_ridge.fit(X_train, y_train)
        best_ridge = grid_ridge.best_estimator_
        y_pred_train_ridge = best_ridge.predict(X_train)
        y_pred_test_ridge = best_ridge.predict(X_test)

        # ------------------ Lasso
        pipe_lasso = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2)),
                ("lasso", Lasso(alpha=0.1, max_iter=10000)),
            ]
        )
        grid_lasso = GridSearchCV(
            pipe_lasso,
            param_grid={"lasso__alpha": np.logspace(-3, 3, 100)},
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_lasso.fit(X_train, y_train)
        best_lasso = grid_lasso.best_estimator_
        y_pred_train_lasso = best_lasso.predict(X_train)
        y_pred_test_lasso = best_lasso.predict(X_test)

        # ------------------ ElasticNet
        pipe_enet = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2)),
                ("enet", ElasticNet(max_iter=10000)),
            ]
        )
        grid_enet = GridSearchCV(
            pipe_enet,
            param_grid={
                "enet__alpha": np.logspace(-3, 3, 100),
                "enet__l1_ratio": np.linspace(0.1, 1, 50),
            },
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_enet.fit(X_train, y_train)
        best_enet = grid_enet.best_estimator_
        y_pred_train_enet = best_enet.predict(X_train)
        y_pred_test_enet = best_enet.predict(X_test)

        # ------------------- Compile results

        results.append(
            {
                "Model": "OLS with Polynomial Features",
                "Train R²": r2_score(y_train, y_pred_train_ols),
                "Test R²": r2_score(y_test, y_pred_test_ols),
                "Train MSE": mean_squared_error(y_train, y_pred_train_ols),
                "Test MSE": mean_squared_error(y_test, y_pred_test_ols),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_ols)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_ols)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_ols),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_ols),
            }
        )
        results.append(
            {
                "Model": f"Ridge (alpha={grid_ridge.best_params_['ridge__alpha']:.4f})",
                "Train R²": r2_score(y_train, y_pred_train_ridge),
                "Test R²": r2_score(y_test, y_pred_test_ridge),
                "Train MSE": mean_squared_error(y_train, y_pred_train_ridge),
                "Test MSE": mean_squared_error(y_test, y_pred_test_ridge),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_ridge)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_ridge),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_ridge),
            }
        )
        results.append(
            {
                "Model": f"Lasso (alpha={grid_lasso.best_params_['lasso__alpha']:.4f})",
                "Train R²": r2_score(y_train, y_pred_train_lasso),
                "Test R²": r2_score(y_test, y_pred_test_lasso),
                "Train MSE": mean_squared_error(y_train, y_pred_train_lasso),
                "Test MSE": mean_squared_error(y_test, y_pred_test_lasso),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_lasso)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_lasso)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_lasso),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_lasso),
            }
        )
        results.append(
            {
                "Model": f"ElasticNet (alpha={grid_enet.best_params_['enet__alpha']:.4f}, l1_ratio={grid_enet.best_params_['enet__l1_ratio']:.2f})",
                "Train R²": r2_score(y_train, y_pred_train_enet),
                "Test R²": r2_score(y_test, y_pred_test_enet),
                "Train MSE": mean_squared_error(y_train, y_pred_train_enet),
                "Test MSE": mean_squared_error(y_test, y_pred_test_enet),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_enet)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_enet)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_enet),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_enet),
            }
        )

        model_objects = [best_ols, best_ridge, best_lasso, best_enet]
        results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True)
        best_model_obj = model_objects[results_df.index[0]]

        return best_ols, best_ridge, best_lasso, best_enet, results_df, best_model_obj

    H = 40  # Test size of 40 samples
    train_lin = df_model.iloc[:-H].copy()
    test_lin = df_model.iloc[-H:].copy()
    # train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
    predictors = [
        "REM hrs",
        "Stress_prev_day",
        "Deep hrs",
        "Wake Count",
        "Sleep_hr_surplus",
        "Respiration",
        "Stress_sleep",
    ]

    time_start = time.time()
    ols, ridge, lasso, enet, results, best_linear = fit_reg_linear_models(
        train_lin[predictors], train_lin["Score"], test_lin[predictors], test_lin["Score"]
    )
    time_end = time.time()
    st.header("📊 Results Dataframe")
    st.dataframe(results)
    st.badge(f"Best model: {results.iloc[0]['Model']}", color="green", icon="✅")
    st.info(f"Time taken to fit models: {time_end - time_start:.2f} seconds")

    # ----------------------------- BEST MODEL PERFORMANCE -----------------------------
    st.header("📊 Performance of Linear Models")
    y_pred_train_best = best_linear.predict(train_lin[predictors])
    y_pred_test_best = best_linear.predict(test_lin[predictors])
    r2_train_linear_pol = r2_score(train_lin["Score"], y_pred_train_best)
    r2_test_linear_pol = r2_score(test_lin["Score"], y_pred_test_best)
    rmse_train_linear_pol = np.sqrt(mean_squared_error(train_lin["Score"], y_pred_train_best))
    rmse_test_linear_pol = np.sqrt(mean_squared_error(test_lin["Score"], y_pred_test_best))
    mae_train_linear_pol = mean_absolute_error(train_lin["Score"], y_pred_train_best)
    mae_test_linear_pol = mean_absolute_error(test_lin["Score"], y_pred_test_best)
    mse_train_linear_pol = rmse_train_linear_pol**2
    mse_test_linear_pol = rmse_test_linear_pol**2

    # ----------------------------- PERFORMANCE METRICS -----------------------------
    with st.expander("📈 Performance Metrics: Train vs Test", expanded=True):
        st.subheader("📈 Train Set Performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Train R²", f"{r2_train_linear_pol:.3f}")
        with c2:
            st.metric("Train MSE", f"{mse_train_linear_pol:.3f}")
        with c3:
            st.metric("Train MAE", f"{mae_train_linear_pol:.3f}")
        with c4:
            st.metric("Train RMSE", f"{rmse_train_linear_pol:.3f}")
        with c5:
            st.metric("Train Samples", f"{train_lin.shape[0]}")
        with c6:
            st.metric("Training Start Date", f"{train_lin.Date.min().date()}")

        st.subheader("📉 Test Set Performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric(
                "Test R²",
                f"{r2_test_linear_pol:.3f}",
                delta=f"{r2_test_linear_pol - r2_train_linear_pol:.3f}",
                delta_color="green" if r2_test_linear_pol > r2_train_linear_pol else "red",
            )
        with c2:
            st.metric(
                "Test MSE",
                f"{mse_test_linear_pol:.3f}",
                delta=f"{mse_test_linear_pol - mse_train_linear_pol:.3f}",
                delta_color="red" if mse_test_linear_pol > mse_train_linear_pol else "green",
                help="Mean Squared Error (MSE): lower values indicate better fit.\
                                Penalizes larger errors more heavily.",
            )
        with c3:
            st.metric(
                "Test MAE",
                f"{mae_test_linear_pol:.3f}",
                delta=f"{mae_test_linear_pol - mae_train_linear_pol:.3f}",
                delta_color="red" if mae_test_linear_pol > mae_train_linear_pol else "green",
                help="Mean Absolute Error (MAE): lower values indicate better fit.",
            )
        with c4:
            st.metric(
                "Test RMSE",
                f"{rmse_test_linear_pol:.3f}",
                delta=f"{rmse_test_linear_pol - rmse_train_linear_pol:.3f}",
                delta_color="red" if rmse_test_linear_pol > rmse_train_linear_pol else "green",
                help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.",
            )
        with c5:
            st.metric(
                "Test Samples",
                f"{test_lin.shape[0]}",
                help="The last 40 samples used for testing.",
            )
        with c6:
            st.metric("Test Start Date", f"{test_lin.Date.min().date()}")
    # ----------------------------- LEARNING CURVE -----------------------------
    with st.expander("📈 Learning Curve Analysis for Best Linear Model", expanded=True):
        st.header("📈 Learning Curve for Best Linear Model")

        def metrics_lcv(df, x_train, x_test, y_train, y_test, model=best_linear):
            """Plot the learning curve for the winning linear model.

            Args:
                df: Full DataFrame.
                x_train: Training features.
                x_test: Test features.
                y_train: Training target.
                y_test: Test target.
                model: Fitted estimator to evaluate. Defaults to the best
                    linear model.

            Returns:
                None. Displays the learning-curve plot.
            """
            from sklearn.base import clone

            train_sizes = np.linspace(10, len(x_train), 10).astype(int)
            train_rmse_list = []
            test_rmse_list = []

            for ts in train_sizes:
                m = clone(model)
                m.fit(x_train.iloc[:ts], y_train.iloc[:ts])
                train_rmse_list.append(
                    np.sqrt(mean_squared_error(y_train.iloc[:ts], m.predict(x_train.iloc[:ts])))
                )
                test_rmse_list.append(np.sqrt(mean_squared_error(y_test, m.predict(x_test))))

            train_mse_mean = np.array(train_rmse_list)
            test_mse_mean = np.array(test_rmse_list)
            lc_df = pd.DataFrame(
                {
                    "Train Size": train_sizes,
                    "Train Score": train_mse_mean,
                    "Test Score": test_mse_mean,
                }
            )
            lc_df["Gap"] = lc_df["Test Score"] - lc_df["Train Score"]

            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            last_step = list(model.named_steps.values())[-1]
            ax[0].plot(train_sizes, train_mse_mean, label="Train RMSE", color="steelblue")
            ax[0].plot(train_sizes, test_mse_mean, label="Test RMSE (holdout)", color="orange")
            ax[0].set_title(
                f"Learning Curve — {last_step.__class__.__name__}",
                fontweight="bold",
                fontsize=8,
                pad=15,
            )
            ax[0].set_xlabel("Training Set Size")
            ax[0].set_ylabel("RMSE")
            sns.despine(ax=ax[0])
            ax[0].legend()
            ax[0].grid()

            sns.barplot(
                x=["Train", "Test"],
                y=[train_mse_mean[-1], test_mse_mean[-1]],
                ax=ax[1],
                palette=["lightblue", "orange"],
            )
            ax[1].set_title(
                "Final RMSE (Full Training Set)",
                fontweight="bold",
                fontsize=8,
                pad=15,
            )
            ax[1].set_ylabel("RMSE")
            sns.despine(ax=ax[1])
            ax[1].grid(axis="y")

            sns.lineplot(
                x=lc_df["Train Size"],
                y=lc_df["Gap"],
                marker="o",
                color="coral",
                label="RMSE gap Train - Test",
                ax=ax[2],
            )
            ax[2].set_title(
                "Gap Between Train and Test RMSE ", fontweight="bold", fontsize=8, pad=15
            )
            ax[2].set_xlabel("Training Set Size")
            ax[2].set_ylabel("RMSE Score Gap (Train - Test)")
            ax[2].annotate(
                text=f"Min Gap =\n{lc_df['Gap'].min():.4f}",
                xy=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()], lc_df["Gap"].min()),
                xytext=(
                    lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()] + 10,
                    lc_df["Gap"].min() + 0.05,
                ),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=6,
                fontweight="bold",
            )
            ax[2].set_ylim(lc_df["Gap"].min() - 0.02, lc_df["Gap"].max() + 0.05)
            sns.despine(ax=ax[2])
            ax[2].legend(loc="upper right")

            st.pyplot(fig)

        metrics_lcv(
            df_model,
            train_lin[predictors],
            test_lin[predictors],
            train_lin["Score"],
            test_lin["Score"],
            model=best_linear,
        )

    # --------------------------- EXPLANATORY POWER -----------------------------
    with st.expander("📊 Explanatory Power of Predictors", expanded=True):
        st.subheader("📊 Explanatory Power of Predictors")
        # Background dataset for SHAP values
        X_background = shap.sample(train_lin[predictors], 100, random_state=42)
        # Cached: SHAP no se recalcula si no cambian los datos.
        shap_values = _compute_shap_values(best_linear, X_background, test_lin[predictors])
        sample_ind = -1  # Last sample in the test set

        force_plot = shap.plots.force(shap_values[sample_ind], matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot for last sample {test_lin.index[sample_ind]}")
        st.pyplot(force_plot)
        plt.close(force_plot)
        col1, col2 = st.columns(2)
        with col1:
            # Beeswarm
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)
            plt.close(fig)
            # bar plot of mean absolute SHAP values
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.bar(shap_values, max_display=14, show=False)
            plt.title("Mean Absolute SHAP Values")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Waterfall for the last sample in the test set
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values[sample_ind], max_display=14, show=False)
            plt.title(f"SHAP Waterfall Plot for last sample {test_lin.index[sample_ind]}")
            st.pyplot(fig)
            plt.close(fig)
