"""Tab Sleep — extraída de streamlit_app.py sin cambiar el comportamiento.

Como en Recovery, render devuelve el DataFrame tal como debe quedar la variable
global tras la tab (None / mismo objeto / copia ordenada por Date). Ninguna tab
posterior usa sleep, pero se conserva el mismo contrato por consistencia.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from app.helpers.plots import plot_line


def render(sleep):
    st.header("😴 Sleep")

    if sleep is None:
        st.info("Upload your clean sleep CSV to see charts.")
        return sleep

    if "Date" not in sleep.columns:
        st.error("Sleep CSV must include a Date column.")
        return sleep

    sleep = sleep.sort_values("Date")
    st.subheader("📋 Sleep table")
    st.dataframe(sleep)

    # Score
    if "Score" in sleep.columns:
        st.subheader("⭐ Sleep Score")
        plot_line(sleep.dropna(subset=["Score"]), "Date", "Score", "Sleep Score over time", "Score")

    # Stages
    stage_cols = [c for c in ["REM hrs", "Light hrs", "Deep hrs"] if c in sleep.columns]
    if stage_cols:
        st.subheader("🧱 Sleep Stages (hrs)")
        df_s = sleep[["Date"] + stage_cols].dropna(subset=["Date"]).copy()

        fig, ax = plt.subplots(figsize=(10, 4))
        bottom = np.zeros(len(df_s))
        for col in stage_cols:
            vals = df_s[col].fillna(0).to_numpy()
            ax.bar(df_s["Date"], vals, bottom=bottom, width=0.8, label=col)
            bottom += vals
        ax.set_title("Sleep stages (stacked hours)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("Hours")
        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        ax.set_axisbelow(True)
        ax.legend()
        st.pyplot(fig)

    # Wake Count
    if "Wake Count" in sleep.columns:
        st.subheader("🌙 Wake Count")
        plot_line(
            sleep.dropna(subset=["Wake Count"]),
            "Date",
            "Wake Count",
            "Wake Count over time",
            "Count",
            marker=None,
            color="purple",
            xlabel="",
            rotate_x=True,
            date_locator=mdates.MonthLocator(interval=2),
            linewidth=0.7,
        )
    # Naps
    left, right = st.columns(2)
    if "Asleep_Nap" in sleep.columns:
        with left:
            st.subheader("💤 Nap Asleep (min)")
            slider_nap2 = st.slider(
                "Select number of days to show for Nap Asleep plot",
                60,
                365,
                365,
                1,
                key="nap_asleep_days",
            )
            recent_date = sleep["Date"].max()
            start_date = recent_date - pd.Timedelta(days=slider_nap2)
            filtered_nap = sleep[
                (sleep["Date"] >= start_date) & (sleep["Date"] <= recent_date)
            ].copy()
            df_plot = filtered_nap.dropna(subset=["Asleep_Nap"])
            roll_avg_nap = df_plot["Asleep_Nap"].rolling(window=7, min_periods=1).mean()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                df_plot["Date"],
                df_plot["Asleep_Nap"],
                color="teal",
                linewidth=1.5,
                label="Nap Asleep",
            )
            ax.plot(
                df_plot["Date"],
                roll_avg_nap,
                marker="o",
                markersize=2,
                color="salmon",
                label="7-day MA",
                linewidth=1,
                linestyle="--",
            )
            ax.set_title("Nap Asleep over time")
            ax.set_ylabel("Minutes")
            ax.tick_params(axis="x", rotation=45)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend()
            sns.despine(ax=ax)
            st.pyplot(fig)

            # Monthly total naps
        # TBD

    else:
        st.info("Column 'InBed_Nap' not found in sleep data.")
    with right:
        st.subheader("🛏️ Nap Asleep (min)")
        if "Asleep_Nap" in sleep.columns:
            slider_nap = st.slider(
                "Select number of days to show for Nap Asleep plot",
                60,
                365,
                365,
                1,
                key="nap_asleep",
            )
            recent_date = sleep["Date"].max()
            start_date = recent_date - pd.Timedelta(days=slider_nap)
            sleep_filtered = sleep[
                (sleep["Date"] >= start_date) & (sleep["Date"] <= recent_date)
            ].copy()
            df_nap = sleep_filtered.dropna(subset=["Asleep_Nap"])[["Date", "Asleep_Nap"]].copy()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(df_nap["Date"], df_nap["Asleep_Nap"], color="teal", width=0.8)
            ax.set_title("Nap Asleep over time")
            ax.set_ylabel("Minutes")
            ax.tick_params(axis="x", rotation=45)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            sns.despine(ax=ax)
            st.pyplot(fig)

            # Monthly total nap inbed
            sleep_monthly = (
                sleep_filtered.set_index("Date").resample("M")["Asleep_Nap"].sum().reset_index()
            )
            st.subheader("🗓️ Monthly Nap Asleep Total (min)")
            plot_line(
                sleep_monthly.dropna(subset=["Asleep_Nap"]),
                "Date",
                "Asleep_Nap",
                "Monthly Nap Asleep Total",
                "Minutes",
                marker="o",
                color="coral",
                xlabel="",
                rotate_x=True,
                date_locator=mdates.MonthLocator(interval=1),
                show_grid=True,
                date_formatter=mdates.DateFormatter("%b-%Y"),
            )
        else:
            st.info("Column 'InBed_Nap' not found in sleep data.")

    return sleep
