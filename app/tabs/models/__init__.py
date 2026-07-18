"""Tab Models — despachador.

render(time_series_analysis) re-lee recovery fresco de st.session_state (aislado
de otras tabs), prepara df_model + predictors y delega según el selectbox `types`
en los submódulos regression/ y unsupervised. Classification aún no tiene rama
(igual que en el monolito original).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.helpers.transforms import string_to_decimal_hours

from . import classification, regression, unsupervised


def render(time_series_analysis):
    st.header("⚙️ Models")
    st.success(f"Overall Time Series Analysis: **{time_series_analysis}**")

    recovery = st.session_state.df_recovery.copy()
    recovery["Date"] = pd.to_datetime(recovery["Date"], errors="coerce")  # Convert to datetime
    recovery["Sleep_need_hrs"] = recovery["Sleep Need"].apply(string_to_decimal_hours)
    recovery["Efficiency"] = recovery["Efficiency"].str.replace("%", "").astype(float)
    recovery["Sleep_hr_surplus"] = recovery["Asleep hrs"] - recovery["Sleep_need_hrs"]

    predictors = [
        "REM hrs",
        "Stress_prev_day",
        "Deep hrs",
        "Wake Count",
        "Sleep_hr_surplus",
        "Respiration",
        "Stress_sleep",
    ]
    df_model = recovery[["Date"] + predictors + ["Score", "Quality"]].dropna().copy()
    df_model = df_model.sort_values("Date")

    st.write(
        "Modeling on: ",
        df_model.shape[0],
        "samples with no missing values in selected features and Score.",
    )

    for col in predictors:
        if col not in df_model.columns:
            st.error(f"Predictor column '{col}' not found in data.")
            st.stop()

    types = st.selectbox(
        "Select algorithm to solve:",
        options=["Regression", "Classification", "Unsupervised"],
        key="algorithm_type_solver",
    )
    if types == "Regression":
        regression.render(df_model, predictors)
    elif types == "Classification":
        classification.render(df_model, predictors)
    # ---------------------------------- UNSUPERVISED LEARNING ----------------------------------
    elif types == "Unsupervised":
        unsupervised.render(df_model)
