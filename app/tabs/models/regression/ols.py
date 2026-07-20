"""Sub-rama 'OLS diagnosis' de Regression (diagnóstico OLS, VIF, residuos).

Extraída de models/regression sin cambiar la lógica. render(df_model, predictors).

Nota: esta sub-rama NO usa SHAP a propósito. El OLS de statsmodels ajusta sobre
los predictores originales (sin transformar), así que sus coeficientes con
p-values, intervalos de confianza y errores robustos HC3 ya son la explicación
exacta; SHAP solo re-derivaría lo mismo (β·(x−x̄)) sin inferencia. SHAP aplica en
las demás sub-ramas (features transformadas o cajas negras) vía shap_utils.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from app.helpers.plots import correlation_insight
from app.helpers.stats import fit_distribution, metrics_learning_curve, normality_test


def render(df_model, predictors):
    """Render the OLS diagnosis sub-branch.

    Fits an OLS model and shows regression diagnostics: VIF, residual
    analysis, heteroskedasticity/specification tests and a learning curve.
    No SHAP by design — see the module docstring.

    Args:
        df_model: Model-ready DataFrame of predictors and target.
        predictors: Predictor column names.

    Returns:
        None.
    """
    # ------------------------------FROZEN MODEL CONDITIONALS-----------------------------
    if "model_frozen" not in st.session_state:
        st.session_state.model_frozen = None
    if "freeze_date" not in st.session_state:
        st.session_state.freeze_date = None
    if "freeze_predictors" not in st.session_state:
        st.session_state.freeze_predictors = None
    n = df_model.shape[0]
    if st.session_state.model_frozen is not None:
        if st.button("Reset frozen model (session)"):
            st.session_state.model_frozen = None
            st.session_state.freeze_date = None
            st.session_state.freeze_predictors = None
            st.rerun()
    # ------------------------------OLS LINEAR REGRESSION TRAINING PHASE-----------------------------
    if (st.session_state.model_frozen is None) and (n < 300):
        st.warning("MODEL ON TRAINING PHASE YET", icon="spinner")

        H = 40  # Test size of 40 samples
        train_lin = df_model.iloc[:-H].copy()
        test_lin = df_model.iloc[-H:].copy()
        # train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
        X = sm.add_constant(train_lin[predictors])
        y = train_lin["Score"]
        model_linear = sm.OLS(y, X).fit(cov_type="HC3")
        X_test = sm.add_constant(test_lin[predictors])
        y_test = test_lin["Score"]
        y_pred_linear = model_linear.predict(X_test)
        r2_train_linear = model_linear.rsquared
        r2_test_linear = r2_score(y_test, y_pred_linear)
        mse_train_linear = mean_squared_error(y, model_linear.fittedvalues)
        mse_test_linear = mean_squared_error(y_test, y_pred_linear)
        mae_train_linear = mean_absolute_error(y, model_linear.fittedvalues)
        mae_test_linear = mean_absolute_error(y_test, y_pred_linear)
        rmse_train_linear = np.sqrt(mse_train_linear)
        rmse_test_linear = np.sqrt(mse_test_linear)
        # --------------------------MODEL DIAGNOSIS--------------------------
        with st.expander("🔎 Model Diagnosis", expanded=False):
            influential_points = st.selectbox(
                "Include Influential Points in Model?",
                [True, False],
                key="influential_points_control",
            )
            c1, c2 = st.columns(2)
            if influential_points:
                with c1:
                    st.subheader("📊 Ramsey RESET Test for Linearity")
                    reset_test = linear_reset(model_linear, power=2, use_f=True)
                    st.write(
                        f"F-statistic: {reset_test.fvalue:.3f}, p-value: {reset_test.pvalue:.3f}"
                    )
                    if reset_test.pvalue < 0.04:
                        st.warning(
                            "Reject the null hypothesis of linearity. Consider adding polynomial or interaction terms.",
                            icon="⚠️",
                        )
                    elif reset_test.pvalue < 0.06:
                        st.info(
                            "Marginal evidence against linearity (p-value close to 0.05). Consider exploring non-linear relationships.",
                            icon="ℹ️",
                        )
                    else:
                        st.success(
                            "Fail to reject the null hypothesis of linearity. No strong evidence of non-linearity detected."
                        )
                    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
                    for ax, col in zip(axes.flatten(), predictors, strict=False):
                        ax.scatter(train_lin[col], y, alpha=0.4, s=20)
                        # Lowess reveals the true shape
                        lowess = sm.nonparametric.lowess(y, train_lin[col], frac=0.6)
                        ax.plot(lowess[:, 0], lowess[:, 1], color="red", linewidth=2)
                        ax.set_title(f"Score vs {col}")
                        sns.despine(ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.subheader("📊 Variance Inflation Factor (VIF)")
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = X.columns
                    vif_data["VIF"] = [
                        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
                    ]
                    vif_data["Meaning"] = vif_data["VIF"].apply(
                        lambda x: "Low multicollinearity"
                        if x < 5
                        else ("Moderate multicollinearity" if x < 10 else "High multicollinearity")
                    )
                    st.dataframe(vif_data)

                    st.subheader("📊 Durbin-Watson Test for Autocorrelation")
                    dw_statistic = durbin_watson(model_linear.resid)
                    st.write(f"Durbin-Watson statistic: {dw_statistic:.3f}")
                    st.info(
                        "Durbin-Watson statistic ranges from 0 to 4. A value around 2 suggests no autocorrelation. Values < 2 indicate positive autocorrelation, while values > 2 indicate negative autocorrelation.",
                        icon="ℹ️",
                    )
                    if dw_statistic < 1.5:
                        st.warning(
                            "Evidence of positive autocorrelation in residuals. Consider adding lagged variables or using time series models.",
                            icon="⚠️",
                        )
                    elif dw_statistic > 2.5:
                        st.warning(
                            "Evidence of negative autocorrelation in residuals. Consider adding lagged variables or using time series models.",
                            icon="⚠️",
                        )
                    else:
                        st.success(
                            "Durbin-Watson statistic close to 2 suggests no strong evidence of autocorrelation in residuals."
                        )

                with c2:
                    st.subheader("📏 Leverage")
                    influence = model_linear.get_influence()
                    leverage = influence.hat_matrix_diag
                    threshold_leverage = 2 * (X.shape[1] + 1) / X.shape[0]
                    high_leverage_points = np.where(leverage > threshold_leverage)[0]

                    st.write(
                        f"\nHigh leverage points (leverage > {threshold_leverage:.4f}): {high_leverage_points}"
                    )
                    st.subheader("📏 Influential Observations (Cook's Distance)")
                    influence = model_linear.get_influence()
                    cooks_d, p_values = influence.cooks_distance

                    threshold = 4 / len(y)
                    influential = np.where(cooks_d > threshold)[0]
                    st.write(
                        f"\nInfluential points (Cook's D > 4/n={threshold:.4f}): {influential}"
                    )

                    # --- Plot ---
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.stem(range(len(cooks_d)), cooks_d, markerfmt=",", basefmt="gray")
                    ax.axhline(
                        threshold,
                        color="red",
                        linestyle="--",
                        label=f"Threshold 4/n = {threshold:.4f}\nAmount of influences: {len(influential)}",
                    )

                    for i in influential:
                        ax.annotate(
                            f"{i}",
                            (i, cooks_d[i]),
                            textcoords="offset points",
                            xytext=(0, 5),
                            fontsize=8,
                            color="red",
                        )
                    ax.set_xlabel("Observation Index")
                    ax.set_ylabel("Cook's Distance")
                    ax.set_title(
                        "Cook's Distance — Influential Observations",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.legend()
                    sns.despine(ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)

                    df_model_influential = df_model.iloc[influential]
                    st.dataframe(
                        df_model_influential[["Date"] + predictors + ["Score"]].reset_index(
                            drop=True
                        )
                    )
            elif not influential_points:
                influence = model_linear.get_influence()
                leverage = influence.hat_matrix_diag
                threshold = 4 / len(y)
                influence = model_linear.get_influence()
                cooks_d, p_values = influence.cooks_distance
                influential = np.where(cooks_d > threshold)[0]
                df_model_no_influential = df_model.drop(df_model.index[influential])
                train_lin = df_model_no_influential.iloc[:-H].copy()
                test_lin = df_model_no_influential.iloc[-H:].copy()
                X = sm.add_constant(train_lin[predictors])
                y = train_lin["Score"]
                model_linear = sm.OLS(y, X).fit(cov_type="HC3")
                r2_train_linear = model_linear.rsquared
                r2_test_linear = r2_score(y_test, y_pred_linear)
                mse_train_linear = mean_squared_error(y, model_linear.fittedvalues)
                mse_test_linear = mean_squared_error(y_test, y_pred_linear)
                mae_train_linear = mean_absolute_error(y, model_linear.fittedvalues)
                mae_test_linear = mean_absolute_error(y_test, y_pred_linear)
                rmse_train_linear = np.sqrt(mse_train_linear)
                rmse_test_linear = np.sqrt(mse_test_linear)
                with c1:
                    st.subheader("📊 Ramsey RESET Test for Linearity")
                    reset_test = linear_reset(model_linear, power=2, use_f=True)
                    st.write(
                        f"F-statistic: {reset_test.fvalue:.3f}, p-value: {reset_test.pvalue:.3f}"
                    )
                    if reset_test.pvalue < 0.04:
                        st.warning(
                            "Reject the null hypothesis of linearity. Consider adding polynomial or interaction terms.",
                            icon="⚠️",
                        )
                    elif reset_test.pvalue < 0.06:
                        st.info(
                            "Marginal evidence against linearity (p-value close to 0.05). Consider exploring non-linear relationships.",
                            icon="ℹ️",
                        )
                    else:
                        st.success(
                            "Fail to reject the null hypothesis of linearity. No strong evidence of non-linearity detected."
                        )
                    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                    for ax, col in zip(axes.flatten(), predictors, strict=False):
                        ax.scatter(train_lin[col], y, alpha=0.4, s=20)
                        # Lowess reveals the true shape
                        lowess = sm.nonparametric.lowess(y, train_lin[col], frac=0.6)
                        ax.plot(lowess[:, 0], lowess[:, 1], color="red", linewidth=2)
                        ax.set_title(f"Score vs {col}")
                        sns.despine(ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.subheader("📊 Variance Inflation Factor (VIF)")
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = X.columns
                    vif_data["VIF"] = [
                        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
                    ]
                    vif_data["Meaning"] = vif_data["VIF"].apply(
                        lambda x: "Low multicollinearity"
                        if x < 5
                        else ("Moderate multicollinearity" if x < 10 else "High multicollinearity")
                    )
                    st.dataframe(vif_data)

                    st.subheader("📊 Durbin-Watson Test for Autocorrelation")
                    dw_statistic = durbin_watson(model_linear.resid)
                    st.write(f"Durbin-Watson statistic: {dw_statistic:.3f}")
                    st.info(
                        "Durbin-Watson statistic ranges from 0 to 4. A value around 2 suggests no autocorrelation. Values < 2 indicate positive autocorrelation, while values > 2 indicate negative autocorrelation.",
                        icon="ℹ️",
                    )
                    if dw_statistic < 1.5:
                        st.warning(
                            "Evidence of positive autocorrelation in residuals. Consider adding lagged variables or using time series models.",
                            icon="⚠️",
                        )
                    elif dw_statistic > 2.5:
                        st.warning(
                            "Evidence of negative autocorrelation in residuals. Consider adding lagged variables or using time series models.",
                            icon="⚠️",
                        )
                    else:
                        st.success(
                            "Durbin-Watson statistic close to 2 suggests no strong evidence of autocorrelation in residuals."
                        )

                with c2:
                    st.warning(
                        f"Removed {len(influential)} influential points from the model.",
                        icon="⚠️",
                    )
                    st.subheader("📏 Leverage")
                    threshold_leverage = 2 * (X.shape[1] + 1) / X.shape[0]
                    high_leverage_points = np.where(leverage > threshold_leverage)[0]
                    st.write(
                        f"\nHigh leverage points (leverage > {threshold_leverage:.4f}): {high_leverage_points}"
                    )

                    st.subheader("📏 Influential Observations (Cook's Distance)")
                    st.write(
                        f"\nInfluential points (Cook's D > 4/n={threshold:.4f}): {influential}"
                    )
                    # --- Plot ---
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.stem(range(len(cooks_d)), cooks_d, markerfmt=",", basefmt="gray")
                    ax.axhline(
                        threshold,
                        color="red",
                        linestyle="--",
                        label=f"Threshold 4/n = {threshold:.4f}\nAmount of influences: {len(influential)}",
                    )

                    for i in influential:
                        ax.annotate(
                            f"{i}",
                            (i, cooks_d[i]),
                            textcoords="offset points",
                            xytext=(0, 5),
                            fontsize=8,
                            color="red",
                        )
                    ax.set_xlabel("Observation Index")
                    ax.set_ylabel("Cook's Distance")
                    ax.set_title(
                        "Cook's Distance — Influential Observations",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.legend()
                    sns.despine(ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
            # ----------------------------- PERFORMANCE METRICS -----------------------------
            st.subheader("📈 Train Set Performance")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Train R²", f"{r2_train_linear:.3f}")
            with c2:
                st.metric("MSE", f"{mse_train_linear:.3f}")
            with c3:
                st.metric("MAE", f"{mae_train_linear:.3f}")
            with c4:
                st.metric("RMSE", f"{rmse_train_linear:.3f}")
            with c5:
                st.metric("Samples", f"{train_lin.shape[0]}")
            with c6:
                st.metric("Training Start Date", f"{train_lin.Date.min().date()}")

            st.subheader("📉 Test Set Performance")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric(
                    "Test R²",
                    f"{r2_test_linear:.3f}",
                    delta=f"{r2_test_linear - r2_train_linear:.3f}",
                    delta_color="green" if r2_test_linear > r2_train_linear else "red",
                )
            with c2:
                st.metric(
                    "MSE",
                    f"{mse_test_linear:.3f}",
                    delta=f"{mse_test_linear - mse_train_linear:.3f}",
                    delta_color="red" if mse_test_linear > mse_train_linear else "green",
                    help="Mean Squared Error (MSE): lower values indicate better fit.\
                            Penalizes larger errors more heavily.",
                )
            with c3:
                st.metric(
                    "MAE",
                    f"{mae_test_linear:.3f}",
                    delta=f"{mae_test_linear - mae_train_linear:.3f}",
                    delta_color="red" if mae_test_linear > mae_train_linear else "green",
                    help="Mean Absolute Error (MAE): lower values indicate better fit.",
                )
            with c4:
                st.metric(
                    "RMSE",
                    f"{rmse_test_linear:.3f}",
                    delta=f"{rmse_test_linear - rmse_train_linear:.3f}",
                    delta_color="red" if rmse_test_linear > rmse_train_linear else "green",
                    help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.",
                )
            with c5:
                st.metric(
                    "Samples",
                    f"{test_lin.shape[0]}",
                    help="The last 40 samples used for testing.",
                )
            with c6:
                st.metric("Test Start Date", f"{test_lin.Date.min().date()}")

        # --------------------------------------OLS INSIGHTS---------------------------------
        predictors = [
            "REM hrs",
            "Stress_prev_day",
            "Deep hrs",
            "Wake Count",
            "Sleep_hr_surplus_centered",
            "Sleep_hr_surplus_squared",
            "Respiration",
            "Stress_sleep",
        ]
        train_lin_pol = df_model.iloc[:-H].copy()
        test_lin_pol = df_model.iloc[-H:].copy()
        surplus_mean = train_lin_pol["Sleep_hr_surplus"].mean()

        train_lin_pol["Sleep_hr_surplus_centered"] = (
            train_lin_pol["Sleep_hr_surplus"] - surplus_mean
        )
        train_lin_pol["Sleep_hr_surplus_squared"] = train_lin_pol["Sleep_hr_surplus_centered"] ** 2

        test_lin_pol["Sleep_hr_surplus_centered"] = test_lin_pol["Sleep_hr_surplus"] - surplus_mean
        test_lin_pol["Sleep_hr_surplus_squared"] = test_lin_pol["Sleep_hr_surplus_centered"] ** 2

        # train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
        X_pol = sm.add_constant(train_lin_pol[predictors])
        y_pol = train_lin_pol["Score"]
        model_linear_pol = sm.OLS(y_pol, X_pol).fit(cov_type="HC3")
        X_test_pol = sm.add_constant(test_lin_pol[predictors])
        y_test_pol = test_lin_pol["Score"]
        y_pred_linear_pol = model_linear_pol.predict(X_test_pol)
        r2_train_linear_pol = model_linear_pol.rsquared
        r2_test_linear_pol = r2_score(y_test_pol, y_pred_linear_pol)
        mse_train_linear_pol = mean_squared_error(y_pol, model_linear_pol.fittedvalues)
        mse_test_linear_pol = mean_squared_error(y_test_pol, y_pred_linear_pol)
        mae_train_linear_pol = mean_absolute_error(y_pol, model_linear_pol.fittedvalues)
        mae_test_linear_pol = mean_absolute_error(y_test_pol, y_pred_linear_pol)
        rmse_train_linear_pol = np.sqrt(mse_train_linear_pol)
        rmse_test_linear_pol = np.sqrt(mse_test_linear_pol)

        # Add polynomial columns to full df_model using train mean (no leakage)
        df_model["Sleep_hr_surplus_centered"] = df_model["Sleep_hr_surplus"] - surplus_mean
        df_model["Sleep_hr_surplus_squared"] = df_model["Sleep_hr_surplus_centered"] ** 2

        df_model["Predicted_Score_Linear_Pol"] = model_linear_pol.predict(
            sm.add_constant(df_model[predictors], has_constant="add")
        )
        df_model["Predicted_Score_Linear_Test_Data_Pol"] = np.nan
        df_model.loc[test_lin_pol.index, "Predicted_Score_Linear_Test_Data_Pol"] = (
            model_linear_pol.predict(sm.add_constant(test_lin_pol[predictors], has_constant="add"))
        )
        df_model["Residuals_Linear_Pol"] = (
            df_model["Score"] - df_model["Predicted_Score_Linear_Pol"]
        )

        # ----------------------------- MODEL SUMMARY & VISUALIZATIONS -----------------------------
        with st.expander("ℹ️ Model Summary", expanded=False):
            c1, c2 = st.columns(2)
            filtered_test = df_model.loc[test_lin_pol.index].dropna(
                subset=["Predicted_Score_Linear_Test_Data_Pol"]
            )
            test_start = filtered_test["Date"].min()
            test_end = filtered_test["Date"].max()
            with c1:
                st.text(model_linear_pol.summary().as_text())
                st.subheader("🔗 Correlations")
                corr = df_model[predictors + ["Score"]].corr()
                st.dataframe(corr)

                for col in corr.columns:
                    if col != "Score":
                        correlation_insight(df_model, "Score", col)
                # ------------------RAMSEY RESET TEST FOR LINEARITY------------------
                reset_test_pol = linear_reset(model_linear_pol, power=2, use_f=True)
                st.subheader("📊 Ramsey RESET Test for Linearity (Polynomial Model)")
                st.write(reset_test_pol)
                st.write(
                    f"F-statistic: {reset_test_pol.fvalue:.3f}, p-value: {reset_test_pol.pvalue:.3f}"
                )
                if reset_test_pol.pvalue < 0.04:
                    st.warning(
                        "Reject the null hypothesis of linearity. Consider adding higher-order polynomial or interaction terms.",
                        icon="⚠️",
                    )
                elif reset_test_pol.pvalue < 0.06:
                    st.info(
                        "Marginal evidence against linearity (p-value close to 0.05). Consider exploring non-linear relationships.",
                        icon="ℹ️",
                    )
                else:
                    st.success(
                        "Fail to reject the null hypothesis of linearity. No strong evidence of non-linearity detected."
                    )
            with c2:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(
                    data=df_model,
                    x="Date",
                    y="Score",
                    label="Actual",
                    ax=ax,
                    color="lightgreen",
                    alpha=0.7,
                )
                sns.lineplot(
                    data=df_model,
                    x="Date",
                    y="Predicted_Score_Linear_Pol",
                    label="Predicted",
                    ax=ax,
                    linewidth=1,
                    color="blue",
                )
                sns.lineplot(
                    data=df_model,
                    x="Date",
                    y="Predicted_Score_Linear_Test_Data_Pol",
                    label="Predicted (Test Data)",
                    ax=ax,
                    linestyle="--",
                    linewidth=1,
                    color="orange",
                )
                ax.axvspan(
                    test_start,
                    test_end,
                    color="lightgrey",
                    alpha=0.2,
                    label="Test Set Period",
                )
                ax.set_title(
                    "Actual vs Predicted Sleep Score (Train & Test Set)",
                    fontweight="bold",
                    fontsize=14,
                    pad=15,
                )
                ax.set_xlabel("")
                ax.set_ylabel("Score")
                ax.tick_params(axis="x", rotation=45)
                ax.legend(loc="best", fontsize=7)
                sns.despine(ax=ax)
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(
                    data=filtered_test,
                    x="Date",
                    y="Score",
                    label="Actual",
                    ax=ax,
                    marker="o",
                    color="green",
                    alpha=0.7,
                )
                sns.lineplot(
                    data=filtered_test,
                    x="Date",
                    y="Predicted_Score_Linear_Test_Data_Pol",
                    label="Predicted (Test Data)",
                    ax=ax,
                    marker="x",
                    linewidth=1.5,
                    linestyle=":",
                    color="orange",
                )
                ax.set_xlabel("")
                ax.set_ylabel("Score")
                ax.set_title(
                    "Actual vs Predicted Sleep Score (Test Set)",
                    fontweight="bold",
                    fontsize=14,
                    pad=15,
                )
                ax.tick_params(axis="x", rotation=45)
                sns.despine(ax=ax)
                ax.legend(loc="lower left", fontsize=7)
                st.pyplot(fig)
                st.markdown(
                    f"**Out-of-sample Test R² (trained on train set):** {r2_test_linear_pol:.3f}"
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(
                    data=filtered_test,
                    x="Score",
                    y="Predicted_Score_Linear_Test_Data_Pol",
                    ax=ax,
                    color="purple",
                    alpha=0.7,
                    hue="Quality",
                    palette="viridis",
                    legend="full",
                )
                sns.lineplot(
                    data=filtered_test,
                    x="Score",
                    y="Score",
                    ax=ax,
                    color="red",
                    linestyle="--",
                    label="Ideal Fit",
                )
                ax.axvline(x=80, color="grey", linestyle=":", label="Good Quality Threshold")
                ax.axvline(x=90, color="grey", linestyle=":", label="Excellent Quality Threshold")
                ax.set_title(
                    "Predicted vs Actual Sleep Score (Test Set)",
                    fontweight="bold",
                    fontsize=14,
                    pad=15,
                )
                ax.set_xlabel("Actual Sleep Score")
                ax.set_ylabel("Predicted Sleep Score")
                sns.despine(ax=ax)
                ax.legend(loc="lower right", fontsize=7)
                st.pyplot(fig)
                # -----------------------------VIF ANALYSIS-----------------------------
                st.subheader("📊 Variance Inflation Factor (VIF) for Polynomial Model")
                vif_data_pol = pd.DataFrame()
                vif_data_pol["feature"] = X_pol.columns
                vif_data_pol["VIF"] = [
                    variance_inflation_factor(X_pol.values, i) for i in range(X_pol.shape[1])
                ]
                vif_data_pol["Meaning"] = vif_data_pol["VIF"].apply(
                    lambda x: "Low multicollinearity"
                    if x < 5
                    else ("Moderate multicollinearity" if x < 10 else "High multicollinearity")
                )
                st.dataframe(vif_data_pol)
                # -----------------------------DURBIN-WATSON TEST-----------------------------
                st.subheader("📊 Durbin-Watson Test for Autocorrelation (Polynomial Model)")
                dw_statistic_pol = durbin_watson(model_linear_pol.resid)
                st.write(f"Durbin-Watson statistic: {dw_statistic_pol:.3f}")
                st.info(
                    "Durbin-Watson statistic ranges from 0 to 4. A value around 2 suggests no autocorrelation. Values < 2 indicate positive autocorrelation, while values > 2 indicate negative autocorrelation.",
                    icon="ℹ️",
                )
                if dw_statistic_pol < 1.5:
                    st.warning(
                        "Evidence of positive autocorrelation in residuals. Consider adding lagged variables or using time series models.",
                        icon="⚠️",
                    )
                elif dw_statistic_pol > 2.5:
                    st.warning(
                        "Evidence of negative autocorrelation in residuals. Consider adding lagged variables or using time series models.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        "Durbin-Watson statistic close to 2 suggests no strong evidence of autocorrelation in residuals."
                    )
            # ----------------------------- PERFORMANCE METRICS -----------------------------
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
                st.metric("Train Samples", f"{train_lin_pol.shape[0]}")
            with c6:
                st.metric("Training Start Date", f"{train_lin_pol.Date.min().date()}")

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
                    f"{test_lin_pol.shape[0]}",
                    help="The last 40 samples used for testing.",
                )
            with c6:
                st.metric("Test Start Date", f"{test_lin_pol.Date.min().date()}")
            # ----------------------------- PREDICTIONS DATAFRAME -----------------------------
            with st.expander("🗂️ Predictions Dataframe", expanded=False):
                st.dataframe(
                    df_model[
                        [
                            "Date",
                            "Score",
                            "Predicted_Score_Linear_Pol",
                            "Predicted_Score_Linear_Test_Data_Pol",
                            "Residuals_Linear_Pol",
                        ]
                    ].sort_values("Date")
                )

            with st.expander("📐 OLS Linear Regression: Insights", expanded=False):
                st.write("Used for prediction:", [f for f in predictors if f in df_model.columns])
            # ----------------------------- LEARNING CURVE ------------------------------------
        with st.expander("📈🧠 Learning Curve & Model Comparison", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Learning Curve Metrics at Key Sample Sizes")
                sample_sizes = range(20, 220, 10)
                results = []
                for sample in sample_sizes:
                    res = metrics_learning_curve(df_model, sample, predictors)
                    if res is not None:
                        results.append(res)

                if results:
                    df_all_metrics = pd.DataFrame(results).set_index("Model_samples")
                    st.dataframe(df_all_metrics)
                else:
                    st.warning("Not enough data to compute learning curve metrics.")
                learning_curve_df = pd.DataFrame(results)

                st.subheader("🪜🧱 Plateaut Detection")
                learning_curve_df = learning_curve_df.set_index("Model_samples")
                learning_curve_df["% Δ RMSE"] = (
                    learning_curve_df["Test RMSE"].diff().fillna(0) * 100
                ) / learning_curve_df["Test RMSE"].shift(1).replace(0, np.nan)
                st.dataframe(learning_curve_df[["Test RMSE", "% Δ RMSE"]])
                st.write("k=", 3)
                st.write("% Δ no greater than", 5)

                # ----------------------------- PLOTTING LEARNING CURVE -----------------------------
            with c2:
                st.subheader("📈 Learning Curve Plot")
                metric = st.selectbox(
                    "Select metric to plot:",
                    ["MAE", "MSE", "RMSE"],
                    index=2,
                    key="learning_curve_metric_selectbox",
                )
                samples = st.checkbox(
                    "Show all values for learning curve (not forecast or extrapolated) ?:",
                    value=True,
                    key="learning_curve_future_values_checkbox",
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                if samples:
                    sns.lineplot(
                        data=learning_curve_df,
                        x=learning_curve_df.index,
                        y=f"Train {metric}",
                        label=f"Train {metric}",
                        ax=ax,
                        color="lightblue",
                        linestyle=":",
                        linewidth=1,
                        marker="o",
                        markersize=4,
                    )
                    sns.lineplot(
                        data=learning_curve_df,
                        x=learning_curve_df.index,
                        y=f"Test {metric}",
                        label=f"Test {metric}",
                        ax=ax,
                        color="orange",
                        linewidth=1,
                        marker="x",
                        markersize=4,
                    )
                    ax.axvspan(
                        xmin=40,
                        xmax=n,
                        color="lightgrey",
                        alpha=0.2,
                        label="Current Region",
                    )
                else:
                    filtered_lc = learning_curve_df.loc[learning_curve_df.index <= n]
                    sns.lineplot(
                        data=filtered_lc,
                        x=filtered_lc.index,
                        y=f"Train {metric}",
                        label=f"Train {metric}",
                        ax=ax,
                        color="lightblue",
                        linestyle=":",
                        linewidth=1,
                        marker="o",
                        markersize=2,
                    )
                    sns.lineplot(
                        data=filtered_lc,
                        x=filtered_lc.index,
                        y=f"Test {metric}",
                        label=f"Test {metric}",
                        ax=ax,
                        color="orange",
                        linewidth=1,
                        marker="x",
                    )

                ax.axvline(x=n, color="white", linestyle="--", label="Current Sample Size")
                ax.set_title(
                    f"Learning Curve: {metric} vs Training Size",
                    fontweight="bold",
                    fontsize=14,
                    pad=15,
                )
                ax.set_xlabel("Model Samples")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=45)
                ax.legend(loc="best", fontsize=7)
                sns.despine(ax=ax)
                st.pyplot(fig)

                for params in model_linear_pol.params.index:
                    if model_linear_pol.pvalues[params] < 0.05:
                        st.success(
                            f"{params} Coeff: {model_linear_pol.params[params]:.4f} P-value: {model_linear_pol.pvalues[params]:.4f} (Significant at α=0.05)"
                        )
                    else:
                        st.warning(
                            f"{params} Coeff: {model_linear_pol.params[params]:.4f} P-value: {model_linear_pol.pvalues[params]:.4f}"
                        )

        # ----------------------------- RESIDUALS ANALYSIS -----------------------------
        with st.expander("📊 Residuals Analysis", expanded=False):
            option_residuals = ["On Train Set", "On Test Set", "Both"]
            residuals_analysis = st.segmented_control(
                "Select residuals analysis dataset:",
                options=option_residuals,
                key="residuals_analysis",
            )
            if residuals_analysis == "On Train Set":
                st.subheader("🧾 Residuals Analysis on Train Set")
                residuals_train_df = df_model[df_model.index.isin(train_lin_pol.index)].copy()
                residuals_train_df["Residuals"] = model_linear_pol.resid
                fitted = model_linear_pol.fittedvalues
                pvalue_resid, interpretation_resid = normality_test(residuals_train_df["Residuals"])
                if pvalue_resid > 0.05:
                    st.success(
                        f"Residuals appear to be normally distributed (p={pvalue_resid:.3f})."
                    )
                else:
                    st.info(
                        f"Residuals do not appear to be normally distributed (p={pvalue_resid:.3f})."
                    )
                st.dataframe(fit_distribution(residuals_train_df["Residuals"]))
                st.subheader(" Breusch-Pagan Test for Heteroscedasticity")
                bp_test = het_breuschpagan(model_linear_pol.resid, model_linear_pol.model.exog)
                bp_labels = [
                    "Lagrange multiplier statistic",
                    "p-value",
                    "f-value",
                    "f p-value",
                ]
                bp_results = dict(zip(bp_labels, bp_test, strict=False))
                if bp_results["p-value"] < 0.05:
                    st.warning(
                        f"❌ Evidence of heteroscedasticity (p={bp_results['p-value']:.3f}). Consider using robust standard errors or transforming variables.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        f"✅ No evidence of heteroscedasticity (p={bp_results['p-value']:.3f})."
                    )
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sm.qqplot(residuals_train_df["Residuals"], line="45", ax=ax, fit=True)
                    ax.set_title(
                        "QQ Plot of Residuals (Train Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(residuals_train_df["Residuals"], kde=True, stat="density", ax=ax)
                    ax.set_title(
                        "Histogram of Residuals (Train Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("Residuals")
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                c3, c4 = st.columns(2)
                with c3:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.scatterplot(x=fitted, y=residuals_train_df["Residuals"], ax=ax)
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title(
                        "Residuals vs Fitted Values (Train Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("Fitted Values")
                    ax.set_ylabel("Residuals")
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                with c4:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.lineplot(data=residuals_train_df, x="Date", y="Residuals", ax=ax)
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title(
                        "Residuals over Time (Train Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("Residuals")
                    ax.tick_params(axis="x", rotation=45)
                    sns.despine(ax=ax)
                    st.pyplot(fig)

            elif residuals_analysis == "On Test Set":
                st.subheader("🧾 Residuals Analysis on Test Set")
                residuals_test_df = df_model[df_model.index.isin(test_lin_pol.index)].copy()
                residuals_test_df["Residuals"] = (
                    residuals_test_df["Score"]
                    - residuals_test_df["Predicted_Score_Linear_Test_Data_Pol"]
                )
                fitted_test = residuals_test_df["Predicted_Score_Linear_Test_Data_Pol"]
                pvalue_resid_test, interpretation_resid_test = normality_test(
                    residuals_test_df["Residuals"]
                )
                if pvalue_resid_test > 0.05:
                    st.success(
                        f"Residuals appear to be normally distributed (p={pvalue_resid_test:.3f})."
                    )
                else:
                    st.info(
                        f"Residuals do not appear to be normally distributed (p={pvalue_resid_test:.3f})."
                    )

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sm.qqplot(residuals_test_df["Residuals"], line="45", ax=ax, fit=True)
                    ax.set_title(
                        "QQ Plot of Residuals (Test Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(residuals_test_df["Residuals"], kde=True, stat="density", ax=ax)
                    ax.set_title(
                        "Histogram of Residuals (Test Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("Residuals")
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                c3, c4 = st.columns(2)
                with c3:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(x=fitted_test, y=residuals_test_df["Residuals"], ax=ax)
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title(
                        "Residuals vs Fitted Values (Test Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("Fitted Values")
                    ax.set_ylabel("Residuals")
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                with c4:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.lineplot(data=residuals_test_df, x="Date", y="Residuals", ax=ax)
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title(
                        "Residuals over Time (Test Set)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("Residuals")
                    ax.tick_params(axis="x", rotation=45)
                    sns.despine(ax=ax)
                    st.pyplot(fig)
            elif residuals_analysis == "Both":
                st.subheader("🧾 Residuals Analysis on Train & Test Set")
                df_model["Residuals"] = df_model["Score"] - df_model["Predicted_Score_Linear_Pol"]
                pvalue_resid_both, interpretation_resid_both = normality_test(df_model["Residuals"])
                if pvalue_resid_both > 0.05:
                    st.success(
                        f"Residuals appear to be normally distributed (p={pvalue_resid_both:.3f})."
                    )
                else:
                    st.info(
                        f"Residuals do not appear to be normally distributed (p={pvalue_resid_both:.3f})."
                    )

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sm.qqplot(df_model["Residuals"], line="45", ax=ax, fit=True)
                    ax.set_title(
                        "QQ Plot of Residuals (All Data)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(df_model["Residuals"], kde=True, stat="density", ax=ax)
                    ax.set_title(
                        "Histogram of Residuals (All Data)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("Residuals")
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                c3, c4 = st.columns(2)
                with c3:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(
                        x=df_model["Predicted_Score_Linear_Pol"],
                        y=df_model["Residuals"],
                        ax=ax,
                    )
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title(
                        "Residuals vs Fitted Values (All Data)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("Fitted Values")
                    ax.set_ylabel("Residuals")
                    sns.despine(ax=ax)
                    st.pyplot(fig)
                with c4:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.lineplot(data=df_model, x="Date", y="Residuals", ax=ax)
                    ax.axhline(0, color="red", linestyle="--")
                    ax.set_title(
                        "Residuals over Time (All Data)",
                        fontsize=14,
                        fontweight="bold",
                        pad=15,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("Residuals")
                    ax.tick_params(axis="x", rotation=45)
                    sns.despine(ax=ax)
                    st.pyplot(fig)

    # ------------------------------FROZEN MODEL DEPLOYMENT PHASE-----------------------------
    elif (st.session_state.model_frozen is None) and (n >= 300):
        st.success("MODEL READY FOR DEPLOYMENT (FREEZING NOW)", icon="✅")
        freeze_date = df_model.iloc[199]["Date"]  # Freeze after first 200 samples (0-199)
        frozen_df = df_model[df_model["Date"] <= freeze_date].copy()

        X_frozen = sm.add_constant(frozen_df[predictors])
        y_frozen = frozen_df["Score"]

        model_frozen = sm.OLS(y_frozen, X_frozen).fit(cov_type="HC3")

        st.session_state.model_frozen = model_frozen
        st.session_state.freeze_date = freeze_date
        st.session_state.freeze_predictors = predictors.copy()

        st.info(
            f"Frozen on {freeze_date.date()} New data after this will be monitored, not used for training.",
            icon="ℹ️",
        )
    # ------------------------------ DEPLOYMENT & MONITORING -----------------------------
    else:
        st.success("MODEL DEPLOYED (SESSION-FROZEN)", icon="✅")
        # Safety: predictors must match
        if st.session_state.freeze_predictors != predictors:
            st.error("Predictors changed after model freeze. Please refresh/reset the model.")
            st.stop()

        model_frozen = st.session_state.model_frozen
        freeze_date = st.session_state.freeze_date

        # Predict for all rows with the frozen model
        X_all = sm.add_constant(df_model[predictors], has_constant="add")
        df_model["yhat_frozen"] = model_frozen.predict(X_all)
        df_model["resid_frozen"] = df_model["Score"] - df_model["yhat_frozen"]

        # Split into in-sample (<= freeze) and live (> freeze)
        live_df = df_model[df_model["Date"] > freeze_date].copy()

        st.caption(f"Frozen on {freeze_date.date()} | Live samples: {len(live_df)}")

        # Show latest prediction
        last = df_model.sort_values("Date").iloc[-1]
        st.metric("Latest predicted score", f"{last['yhat_frozen']:.2f}")

        # Monitoring metrics on LIVE (post-freeze)
        if len(live_df) >= 5:
            mae_live = mean_absolute_error(live_df["Score"], live_df["yhat_frozen"])
            rmse_live = np.sqrt(mean_squared_error(live_df["Score"], live_df["yhat_frozen"]))
            bias_live = live_df["resid_frozen"].mean()

            c1, c2, c3 = st.columns(3)
            c1.metric("Live MAE", f"{mae_live:.2f}")
            c2.metric("Live RMSE", f"{rmse_live:.2f}")
            c3.metric("Live Bias", f"{bias_live:.2f}")
        else:
            st.info("Not enough post-freeze samples yet for stable monitoring.")

        # Plot live residual drift
        with st.expander("📊 Monitoring: Live residuals", expanded=False):
            if len(live_df) > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=live_df, x="Date", y="resid_frozen", ax=ax)
                ax.axhline(0, color="red", linestyle="--")
                ax.set_title("Post-freeze residuals over time")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)
