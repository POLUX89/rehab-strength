"""Tab Recovery — extraída de streamlit_app.py sin cambiar el comportamiento.

render() devuelve el DataFrame recovery tal como debe quedar la variable global
tras la tab: ordenado por Date (una copia) cuando aplica, o el mismo objeto sin
tocar en los casos guardados. Esa copia es la que protege
st.session_state.df_recovery de las mutaciones in-place de las tabs siguientes,
así que se conserva el mismo contrato que tenía el monolito.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import streamlit as st

from app.helpers.plots import plot_line

COMPONENT_CANDIDATES = [
    "Stress_prev_day",
    "Overnight HRV",
    "Resting Heart Rate",
    "Score",
    "RECOVERY_SCORE_RAW",
]


def render(recovery):
    st.header("🧠 Recovery")

    if recovery is None:
        st.info("Upload your clean recovery CSV to see charts.")
        return recovery

    if "Date" not in recovery.columns:
        st.error("Recovery CSV must include a Date column.")
        return recovery

    recovery = recovery.sort_values("Date")

    st.subheader("📋 Recovery table")
    st.dataframe(recovery)

    # Main recovery score (sigmoid)
    if "Sigmoid Recovery Score" in recovery.columns:
        st.subheader("🧠 Sigmoid Recovery Score (0–1)")
        plot_line(
            recovery.dropna(subset=["Sigmoid Recovery Score"]),
            "Date",
            "Sigmoid Recovery Score",
            "Sigmoid Recovery Score over time",
            "Score",
            xlabel="",
            color="seagreen",
            rotate_x=True,
            date_locator=mdates.DayLocator(interval=2),
            date_formatter=mdates.DateFormatter("%b-%d"),
        )

    # Components (choose what you want)
    st.subheader("🧩 Components")
    available = [c for c in COMPONENT_CANDIDATES if c in recovery.columns]
    if available:
        chosen = st.multiselect("Pick component(s) to plot:", available, default=available[:3])
        for col in chosen:
            plot_line(recovery.dropna(subset=[col]), "Date", col, f"{col} over time", col)
    else:
        st.info("No component columns detected (Stress_prev_day / Overnight HRV / etc.).")

    return recovery
