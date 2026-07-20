"""Sub-rama 'Non Linear Models' de Regression (DT/KNN/SVR).

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.tabs.models.shap_utils import compute_shap_values


def render(df_model, predictors):
    """Render the Non Linear Models sub-branch.

    Trains Decision Tree, SVM and KNN (with scaling and grid-searched
    hyperparameters), compares them, and shows metrics and a learning curve
    for the winner.

    Args:
        df_model: Model-ready DataFrame of predictors and target.
        predictors: Predictor column names.

    Returns:
        None.
    """

    @st.cache_data(show_spinner="Fitting non-linear models... (runs once per dataset)")
    def fit_reg_non_linear_models(X_train, y_train, X_test, y_test):
        """Fit Decision Tree, SVM and KNN with scaling and grid-searched hyperparameters.

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
        # ------------------DT------------------
        full_tree = DecisionTreeRegressor()
        path = full_tree.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas
        samples_ccp = np.quantile(ccp_alphas, np.linspace(0, 1, 10))
        pipe_dt = Pipeline([("dt", DecisionTreeRegressor(criterion="squared_error"))])
        grid_dt = GridSearchCV(
            pipe_dt,
            param_grid={
                "dt__max_depth": [3, 5, 7, 9, None],
                "dt__min_samples_split": [2, 5, 10],
                "dt__ccp_alpha": samples_ccp,
            },
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_dt.fit(X_train, y_train)
        best_dt = grid_dt.best_estimator_
        y_pred_train_dt = best_dt.predict(X_train)
        y_pred_test_dt = best_dt.predict(X_test)

        # ------------------ KNN ---------------------------
        pipe_knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())])
        grid_knn = GridSearchCV(
            pipe_knn,
            param_grid={
                "knn__n_neighbors": [1, 3, 5, 7, 10, 15, 20, 30],
                "knn__weights": ["uniform", "distance"],
                "knn__p": [1, 2],
            },
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_knn.fit(X_train, y_train)
        best_knn = grid_knn.best_estimator_
        y_pred_train_knn = best_knn.predict(X_train)
        y_pred_test_knn = best_knn.predict(X_test)

        # ------------------ SVM ---------------------------

        pipe_svmr = Pipeline([("scaler", StandardScaler()), ("svmr", SVR())])
        grid_svmr = GridSearchCV(
            pipe_svmr,
            param_grid=[
                {  # linear — no gamma, no degree
                    "svmr__kernel": ["linear"],
                    "svmr__C": np.logspace(-3, 3, 30),
                },
                {  # rbf and sigmoid — gamma applies, degree does not
                    "svmr__kernel": ["rbf", "sigmoid"],
                    "svmr__C": np.logspace(-3, 3, 30),
                    "svmr__gamma": ["scale", "auto"],
                },
                {  # poly — both gamma and degree apply
                    "svmr__kernel": ["poly"],
                    "svmr__C": np.logspace(-3, 3, 10),
                    "svmr__gamma": ["scale", "auto"],
                    "svmr__degree": [2, 3],
                },
            ],
            cv=5,
            scoring="neg_mean_squared_error",
        )
        grid_svmr.fit(X_train, y_train)
        best_svmr = grid_svmr.best_estimator_
        y_pred_train_svmr = best_svmr.predict(X_train)
        y_pred_test_svmr = best_svmr.predict(X_test)

        # ------------------- Compile results
        results.append(
            {
                "Model": f"Decision Tree (ccp_alpha={grid_dt.best_params_['dt__ccp_alpha']}), max_depth={grid_dt.best_params_['dt__max_depth']}, min_samples_split={grid_dt.best_params_['dt__min_samples_split']}",
                "Train R²": r2_score(y_train, y_pred_train_dt),
                "Test R²": r2_score(y_test, y_pred_test_dt),
                "Train MSE": mean_squared_error(y_train, y_pred_train_dt),
                "Test MSE": mean_squared_error(y_test, y_pred_test_dt),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_dt)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_dt)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_dt),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_dt),
            }
        )
        results.append(
            {
                "Model": f"KNN (n_neighbors={grid_knn.best_params_['knn__n_neighbors']}, weights={grid_knn.best_params_['knn__weights']}, p={grid_knn.best_params_['knn__p']})",
                "Train R²": r2_score(y_train, y_pred_train_knn),
                "Test R²": r2_score(y_test, y_pred_test_knn),
                "Train MSE": mean_squared_error(y_train, y_pred_train_knn),
                "Test MSE": mean_squared_error(y_test, y_pred_test_knn),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_knn)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_knn)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_knn),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_knn),
            }
        )
        results.append(
            {
                "Model": f"SVM Regressor (kernel={grid_svmr.best_params_['svmr__kernel']}",
                "Train R²": r2_score(y_train, y_pred_train_svmr),
                "Test R²": r2_score(y_test, y_pred_test_svmr),
                "Train MSE": mean_squared_error(y_train, y_pred_train_svmr),
                "Test MSE": mean_squared_error(y_test, y_pred_test_svmr),
                "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_svmr)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_svmr)),
                "Train MAE": mean_absolute_error(y_train, y_pred_train_svmr),
                "Test MAE": mean_absolute_error(y_test, y_pred_test_svmr),
            }
        )

        model_objects = [best_dt, best_knn, best_svmr]
        results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True)
        best_model_obj = model_objects[results_df.index[0]]

        return best_dt, best_knn, best_svmr, results_df, best_model_obj

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
    dt, knn, svmr, results, best_model_non_linear = fit_reg_non_linear_models(
        train_lin[predictors], train_lin["Score"], test_lin[predictors], test_lin["Score"]
    )
    time_end = time.time()
    st.header("📊 Results Dataframe")
    st.dataframe(results)
    st.badge(f"Best model: {results.iloc[0]['Model']}", color="green", icon="✅")
    st.badge(f"Best parameters: {best_model_non_linear.get_params()}", icon="✅")
    st.info(f"Time taken to fit models: {time_end - time_start:.2f} seconds")

    # ----------------------------- BEST MODEL PERFORMANCE -----------------------------
    st.header("📊 Performance of Non-Linear Models")
    y_pred_train_best_nonlinear = best_model_non_linear.predict(train_lin[predictors])
    y_pred_test_best_nonlinear = best_model_non_linear.predict(test_lin[predictors])
    r2_train_nonlinear = r2_score(train_lin["Score"], y_pred_train_best_nonlinear)
    r2_test_nonlinear = r2_score(test_lin["Score"], y_pred_test_best_nonlinear)
    rmse_train_nonlinear = np.sqrt(
        mean_squared_error(train_lin["Score"], y_pred_train_best_nonlinear)
    )
    rmse_test_nonlinear = np.sqrt(mean_squared_error(test_lin["Score"], y_pred_test_best_nonlinear))
    mae_train_nonlinear = mean_absolute_error(train_lin["Score"], y_pred_train_best_nonlinear)
    mae_test_nonlinear = mean_absolute_error(test_lin["Score"], y_pred_test_best_nonlinear)
    mse_train_nonlinear = rmse_train_nonlinear**2
    mse_test_nonlinear = rmse_test_nonlinear**2
    # ----------------------------- PERFORMANCE METRICS -----------------------------
    with st.expander("📈 Performance Metrics: Train vs Test", expanded=True):
        st.subheader("📈 Train Set Performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Train R²", f"{r2_train_nonlinear:.3f}")
        with c2:
            st.metric("Train MSE", f"{mse_train_nonlinear:.3f}")
        with c3:
            st.metric("Train MAE", f"{mae_train_nonlinear:.3f}")
        with c4:
            st.metric("Train RMSE", f"{rmse_train_nonlinear:.3f}")
        with c5:
            st.metric("Train Samples", f"{train_lin.shape[0]}")
        with c6:
            st.metric("Training Start Date", f"{train_lin.Date.min().date()}")

        st.subheader("📉 Test Set Performance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric(
                "Test R²",
                f"{r2_test_nonlinear:.3f}",
                delta=f"{r2_test_nonlinear - r2_train_nonlinear:.3f}",
                delta_color="green" if r2_test_nonlinear > r2_train_nonlinear else "red",
            )
        with c2:
            st.metric(
                "Test MSE",
                f"{mse_test_nonlinear:.3f}",
                delta=f"{mse_test_nonlinear - mse_train_nonlinear:.3f}",
                delta_color="red" if mse_test_nonlinear > mse_train_nonlinear else "green",
                help="Mean Squared Error (MSE): lower values indicate better fit.\
                                Penalizes larger errors more heavily.",
            )
        with c3:
            st.metric(
                "Test MAE",
                f"{mae_test_nonlinear:.3f}",
                delta=f"{mae_test_nonlinear - mae_train_nonlinear:.3f}",
                delta_color="red" if mae_test_nonlinear > mae_train_nonlinear else "green",
                help="Mean Absolute Error (MAE): lower values indicate better fit.",
            )
        with c4:
            st.metric(
                "Test RMSE",
                f"{rmse_test_nonlinear:.3f}",
                delta=f"{rmse_test_nonlinear - rmse_train_nonlinear:.3f}",
                delta_color="red" if rmse_test_nonlinear > rmse_train_nonlinear else "green",
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
    with st.expander("📈 Learning Curve Analysis for Best Non Linear Model", expanded=True):
        st.header("📈 Learning Curve for Best Non Linear Model")

        def metrics_lcv_non_linear(
            df, x_train, x_test, y_train, y_test, model=best_model_non_linear
        ):
            """Plot the learning curve for the winning non-linear model.

            Args:
                df: Full DataFrame.
                x_train: Training features.
                x_test: Test features.
                y_train: Training target.
                y_test: Test target.
                model: Fitted estimator to evaluate. Defaults to the best
                    non-linear model.

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

        metrics_lcv_non_linear(
            df_model,
            train_lin[predictors],
            test_lin[predictors],
            train_lin["Score"],
            test_lin["Score"],
            model=best_model_non_linear,
        )
    # --------------------------- EXPLANATORY POWER -----------------------------
    with st.expander("📊 Explanatory Power of Predictors", expanded=True):
        st.subheader("📊 Explanatory Power of Predictors")
        # Background dataset — small sample is enough for speed
        X_background = shap.sample(train_lin[predictors], 100, random_state=42)
        # Cached; TreeExplainer if the winner is a tree, else model-agnostic.
        shap_values_non_linear = compute_shap_values(
            best_model_non_linear, X_background, test_lin[predictors], cache_key="reg_nonlinear"
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
