"""Sub-rama 'Bagging & Boosting Models' de Regression (RF/AdaBoost/GradientBoosting).

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
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.tabs.models.shap_utils import compute_shap_values


def render(df_model, predictors):
    """Render the Bagging & Boosting sub-branch.

    Trains Random Forest, AdaBoost and Gradient Boosting with grid search,
    compares them, and shows metrics and a learning curve for the winner.

    Args:
        df_model: Model-ready DataFrame of predictors and target.
        predictors: Predictor column names.

    Returns:
        None.
    """

    @st.cache_data(show_spinner="Training ensemble models... (runs once per dataset)")
    def fit_ensemble_models(X_train, y_train, X_test, y_test):
        """Fit RF, AdaBoost and Gradient Boosting with grid-searched hyperparameters.

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
        # ------------------ RF ---------------------------

        pipe_rf = Pipeline([("rf", RandomForestRegressor(criterion="squared_error", n_jobs=4))])
        grid_rf = GridSearchCV(
            pipe_rf,
            param_grid={
                "rf__n_estimators": [100, 200, 300],
                "rf__max_depth": [None, 10, 20, 30],
                "rf__min_samples_leaf": [2, 5, 10],
                "rf__max_features": ["sqrt", "log2", 0.33, 0.5],
            },
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_rf.fit(X_train, y_train)
        best_rf = grid_rf.best_estimator_
        y_pred_train_rf = best_rf.predict(X_train)
        y_pred_test_rf = best_rf.predict(X_test)

        # ------------------ AdaBoost ---------------------------
        param_grid = [
            # DT — no scaling needed
            {
                "ada__estimator": [DecisionTreeRegressor()],
                "ada__estimator__max_depth": [1, 2, 3],  # DT hyperparameter
                "ada__estimator__min_samples_leaf": [1, 5, 10],  # DT hyperparameter
                "ada__n_estimators": [100, 200, 300],
                "ada__learning_rate": [0.01, 0.1, 0.5],
            },
            # SVR — scaling required
            {
                "ada__estimator": [SVR()],
                "ada__estimator__C": [0.1, 1.0, 10.0],  # SVR Hyperparameter
                "ada__estimator__kernel": ["rbf", "linear"],  # SVR Hyperparameter
                "ada__n_estimators": [100, 200],  # Adaboost
                "ada__learning_rate": [0.01, 0.1],  # Adaboost
            },
        ]

        pipe = Pipeline([("scaler", StandardScaler()), ("ada", AdaBoostRegressor())])

        grid_ada = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,  # note: now uses "ada__" prefix
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=4,
        )
        grid_ada.fit(X_train, y_train)
        best_ada = grid_ada.best_estimator_
        y_pred_train_ada = best_ada.predict(X_train)
        y_pred_test_ada = best_ada.predict(X_test)

        # ------------------ Gradient Boosting ---------------------------
        pipe_gb = Pipeline([("gb", GradientBoostingRegressor(loss="squared_error"))])
        grid_gb = GridSearchCV(
            pipe_gb,
            param_grid={
                "gb__learning_rate": np.logspace(-3, 0, 10),
                "gb__n_estimators": [100, 200, 300],
                "gb__max_depth": [3, 4, 5],
                # None = all features (the old "auto", removed in sklearn 1.3+).
                "gb__max_features": [None, "sqrt", "log2"],
                "gb__subsample": [0.6, 0.8, 1.0],
            },
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=4,
        )
        grid_gb.fit(X_train, y_train)
        best_gb = grid_gb.best_estimator_
        y_pred_train_gb = best_gb.predict(X_train)
        y_pred_test_gb = best_gb.predict(X_test)

        # ------------------- Compile results ---------------------------
        results.append(
            {
                "Model": f"Random Forest (n_estimators={grid_rf.best_params_['rf__n_estimators']}, max_depth={grid_rf.best_params_['rf__max_depth']}, min_samples_leaf={grid_rf.best_params_['rf__min_samples_leaf']}, max_features={grid_rf.best_params_['rf__max_features']})",
                "Train R²": r2_score(y_train, y_pred_train_rf),
                "Test R²": r2_score(y_test, y_pred_test_rf),
                "Train MSE": mean_squared_error(y_train, y_pred_train_rf),
                "Test MSE": mean_squared_error(y_test, y_pred_test_rf),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_rf)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_rf),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_rf),
            }
        )
        results.append(
            {
                "Model": f"AdaBoost (estimator={grid_ada.best_params_['ada__estimator'].__class__.__name__}, n_estimators={grid_ada.best_params_['ada__n_estimators']}, learning_rate={grid_ada.best_params_['ada__learning_rate']})",
                "Train R²": r2_score(y_train, y_pred_train_ada),
                "Test R²": r2_score(y_test, y_pred_test_ada),
                "Train MSE": mean_squared_error(y_train, y_pred_train_ada),
                "Test MSE": mean_squared_error(y_test, y_pred_test_ada),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_ada)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_ada)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_ada),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_ada),
            }
        )
        results.append(
            {
                "Model": f"Gradient Boosting (n_estimators={grid_gb.best_params_['gb__n_estimators']}, learning_rate={grid_gb.best_params_['gb__learning_rate']}, max_depth={grid_gb.best_params_['gb__max_depth']}, max_features={grid_gb.best_params_['gb__max_features']}, subsample={grid_gb.best_params_['gb__subsample']})",
                "Train R²": r2_score(y_train, y_pred_train_gb),
                "Test R²": r2_score(y_test, y_pred_test_gb),
                "Train MSE": mean_squared_error(y_train, y_pred_train_gb),
                "Test MSE": mean_squared_error(y_test, y_pred_test_gb),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_gb)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_gb)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_gb),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_gb),
            }
        )
        model_objects = [best_rf, best_ada, best_gb]
        results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True)
        best_model_obj = model_objects[results_df.index[0]]

        return best_rf, best_ada, best_gb, results_df, best_model_obj

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
    rf, ada, gb, results, best_model_ensemble = fit_ensemble_models(
        train_lin[predictors], train_lin["Score"], test_lin[predictors], test_lin["Score"]
    )
    time_end = time.time()
    st.header("📊 Results Dataframe")
    st.dataframe(results)
    st.badge(f"Best model: {results.iloc[0]['Model']}", color="green", icon="✅")
    st.badge(f"Best parameters: {best_model_ensemble.get_params()}", icon="✅")
    st.info(f"Time taken to fit models: {time_end - time_start:.2f} seconds")

    # ----------------------------- BEST MODEL PERFORMANCE -----------------------------
    st.header("📊 Performance of Ensemble Models")
    y_pred_train_best_ensemble = best_model_ensemble.predict(train_lin[predictors])
    y_pred_test_best_ensemble = best_model_ensemble.predict(test_lin[predictors])
    r2_train_ensemble = r2_score(train_lin["Score"], y_pred_train_best_ensemble)
    r2_test_ensemble = r2_score(test_lin["Score"], y_pred_test_best_ensemble)
    rmse_train_ensemble = np.sqrt(
        mean_squared_error(train_lin["Score"], y_pred_train_best_ensemble)
    )
    rmse_test_ensemble = np.sqrt(mean_squared_error(test_lin["Score"], y_pred_test_best_ensemble))
    mae_train_ensemble = mean_absolute_error(train_lin["Score"], y_pred_train_best_ensemble)
    mae_test_ensemble = mean_absolute_error(test_lin["Score"], y_pred_test_best_ensemble)
    mse_train_ensemble = rmse_train_ensemble**2
    mse_test_ensemble = rmse_test_ensemble**2

    # ----------------------------- PERFORMANCE METRICS -----------------------------
    with st.expander("📈 Performance Metrics: Train vs Test", expanded=True):
        st.subheader("📈 Train Set Performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Train R²", f"{r2_train_ensemble:.3f}")
        with c2:
            st.metric("Train MSE", f"{mse_train_ensemble:.3f}")
        with c3:
            st.metric("Train MAE", f"{mae_train_ensemble:.3f}")
        with c4:
            st.metric("Train RMSE", f"{rmse_train_ensemble:.3f}")
        with c5:
            st.metric("Train Samples", f"{train_lin.shape[0]}")
        with c6:
            st.metric("Training Start Date", f"{train_lin.Date.min().date()}")

        st.subheader("📉 Test Set Performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric(
                "Test R²",
                f"{r2_test_ensemble:.3f}",
                delta=f"{r2_test_ensemble - r2_train_ensemble:.3f}",
                delta_color="green" if r2_test_ensemble > r2_train_ensemble else "red",
            )
        with c2:
            st.metric(
                "Test MSE",
                f"{mse_test_ensemble:.3f}",
                delta=f"{mse_test_ensemble - mse_train_ensemble:.3f}",
                delta_color="red" if mse_test_ensemble > mse_train_ensemble else "green",
                help="Mean Squared Error (MSE): lower values indicate better fit.\
                                Penalizes larger errors more heavily.",
            )
        with c3:
            st.metric(
                "Test MAE",
                f"{mae_test_ensemble:.3f}",
                delta=f"{mae_test_ensemble - mae_train_ensemble:.3f}",
                delta_color="red" if mae_test_ensemble > mae_train_ensemble else "green",
                help="Mean Absolute Error (MAE): lower values indicate better fit.",
            )
        with c4:
            st.metric(
                "Test RMSE",
                f"{rmse_test_ensemble:.3f}",
                delta=f"{rmse_test_ensemble - rmse_train_ensemble:.3f}",
                delta_color="red" if rmse_test_ensemble > rmse_train_ensemble else "green",
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
    with st.expander("📈 Learning Curve Analysis for Best Ensemble Model", expanded=True):
        st.header("📈 Learning Curve for Best Ensemble Model")

        def metrics_lcv_ensemble(df, x_train, x_test, y_train, y_test, model=best_model_ensemble):
            """Plot the learning curve for the winning ensemble model.

            Args:
                df: Full DataFrame.
                x_train: Training features.
                x_test: Test features.
                y_train: Training target.
                y_test: Test target.
                model: Fitted estimator to evaluate. Defaults to the best
                    ensemble model.

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

        metrics_lcv_ensemble(
            df_model,
            train_lin[predictors],
            test_lin[predictors],
            train_lin["Score"],
            test_lin["Score"],
            model=best_model_ensemble,
        )
    # --------------------------- EXPLANATORY POWER -----------------------------
    with st.expander("📊 Explanatory Power of Predictors", expanded=True):
        st.subheader("📊 Explanatory Power of Predictors")
        # Background dataset — small sample is enough for speed
        X_background = shap.sample(train_lin[predictors], 100, random_state=42)
        # Cached; TreeExplainer if the winner is a tree, else model-agnostic.
        shap_values_non_linear = compute_shap_values(
            best_model_ensemble, X_background, test_lin[predictors]
        )
        sample_ind = -1  # last sample in the test set

        force_plot = shap.plots.force(
            shap_values_non_linear[sample_ind], matplotlib=True, show=False
        )
        plt.title(f"SHAP Force Plot for last sample {test_lin.index[sample_ind]}")
        st.pyplot(force_plot)
        plt.close(force_plot)

        col1, col2 = st.columns(2)
        with col1:
            # Beeswarm
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.beeswarm(shap_values_non_linear, show=False)
            st.pyplot(fig)
            plt.close(fig)
            # Mean absolute SHAP values
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.bar(shap_values_non_linear, max_display=14, show=False)
            plt.title("Mean Absolute SHAP Values")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            # Waterfall for the last sample in the test set
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_values_non_linear[sample_ind], max_display=14, show=False)
            plt.title(f"SHAP Waterfall Plot for last sample {test_lin.index[sample_ind]}")
            st.pyplot(fig)
            plt.close(fig)
