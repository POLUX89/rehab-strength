"""Helpers de visualización (matplotlib + Streamlit).

Extraídos de streamlit_app.py sin cambiar la lógica. A diferencia de stats/
transforms, estas funciones renderizan directamente en Streamlit (st.pyplot,
st.success/info/warning), así que dependen de streamlit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_line(
    dfx,
    x,
    y,
    title,
    ylabel,
    xlabel="Date",
    marker="o",
    markersize=4,
    color=None,
    show_grid=True,
    despine=True,
    rotate_x=False,
    date_locator=None,
    date_formatter=None,
    linewidth=1.5,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfx[x], dfx[y], marker=marker, markersize=markersize, color=color, linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_grid:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    if rotate_x:
        ax.tick_params(axis="x", rotation=45)

    if date_locator:
        ax.xaxis.set_major_locator(date_locator)
    if date_formatter:
        ax.xaxis.set_major_formatter(date_formatter)
    if despine:
        sns.despine(ax=ax)

    st.pyplot(fig)


def plot_two_axis(dfx, x, y1, y2, title, y1_label, y2_label):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(dfx[x], dfx[y1], marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(y1_label)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(dfx[x], dfx[y2], marker="o", linestyle="--")
    ax2.set_ylabel(y2_label)

    ax1.set_title(title)
    st.pyplot(fig)


def correlation_insight(df, col1, col2):
    """Provide insight on correlation between two columns."""
    if df is None or col1 not in df.columns or col2 not in df.columns:
        return "Insufficient data for correlation analysis."
    corr_coef = df[[col1, col2]].dropna().corr().iloc[0, 1]
    if corr_coef == 1:
        return st.success(f"Perfect positive correlation (1.00) between {col1} and {col2}.")
    elif corr_coef > 0.7:
        return st.success(
            f"Strong positive correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef > 0.49:
        return st.info(
            f"Moderate positive correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef > 0:
        return st.warning(
            f"Weak or no significant correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef == -1:
        return st.success(f"Perfect negative correlation (1.00) between {col1} and {col2}.")
    elif corr_coef < -0.7:
        return st.success(
            f"Strong negative correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef < -0.49:
        return st.info(
            f"Moderate negative correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    else:
        return st.warning(
            f"Weak or no significant correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
