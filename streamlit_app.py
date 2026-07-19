# app.py
# Rehab Strength Dashboard (Workouts + Sleep + Recovery)
# Run: streamlit run app.py

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from statsmodels import tsa
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import io
from datetime import date
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    classification_report,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import shap
import time

from app.helpers.stats import (
    compute_ecdf,
    fit_distribution,
    metrics_learning_curve,
    normality_test,
    outlier_dectection_iqr,
    outlier_detection_zscore_modified,
)
from app.helpers.transforms import (
    daily_ma,
    normalize_recovery,
    normalize_sleep,
    normalize_workouts,
    pick_col,
    recovery_zone,
    safe_minimal_last,
    string_to_decimal_hours,
    week_bounds,
    weekly_bucket,
)
from app.helpers.plots import correlation_insight, plot_line
from app.tabs import recovery as recovery_tab
from app.tabs import workouts as workouts_tab
from app.tabs import sleep as sleep_tab
from app.tabs import timeseries as timeseries_tab
from app.tabs import stats as stats_tab
from app.tabs import home as home_tab
from app.tabs import models as models_tab

st.set_page_config(page_title="Rehab Strength APP", layout="wide")
st.title("🏋️‍♂️ Rehab Strength APP", text_alignment="center")
st.caption("Workouts (Strong) • Sleep (Sheets) • Recovery (Sigmoid)")
app_version = "V2.7.0"
st.caption(f"App Version: {app_version} • Updated: {datetime.now():%Y-%m-%d %H:%M}")
st.markdown("---")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Settings")
cva_dt = st.sidebar.date_input("CVA split date", value=datetime(2025, 5, 14))
smooth_days = st.sidebar.slider("Smoothing window (days)", 3, 30, 15, 1)
show_dark = st.sidebar.toggle("🌙 Dark mode", value=True)

if show_dark:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { background-color:#0e1117 !important; color:#e5e7eb !important; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "savefig.facecolor": "#0e1117",
            "text.color": "#e5e7eb",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#e5e7eb",
            "ytick.color": "#e5e7eb",
            "axes.edgecolor": "#e5e7eb",
            "grid.color": "#2d3748",
        }
    )
else:
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)

# -------------------------
# Helpers
# -------------------------


# -------------------------
# Uploads (hidden after loaded)
# -------------------------


# ---------- helpers ----------
def all_loaded() -> bool:
    """Report whether all three datasets are loaded into session state.

    Returns:
        True if ``df_workouts``, ``df_sleep`` and ``df_recovery`` are all
        present and non-None, else False.
    """
    return all(
        st.session_state.get(k) is not None for k in ["df_workouts", "df_sleep", "df_recovery"]
    )


def load_df_from_upload(uploaded_file):
    """Read an uploaded CSV file into a DataFrame.

    Args:
        uploaded_file: A Streamlit ``UploadedFile`` holding CSV bytes.

    Returns:
        The parsed DataFrame.
    """
    data = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(data))


# ---------- init state ----------
if "show_uploads" not in st.session_state:
    st.session_state.show_uploads = True

st.subheader("📥 Upload your cleaned CSVs")

# status + button
left, right = st.columns([3, 1])
with left:
    w_ok = "✅" if st.session_state.get("df_workouts") is not None else "⬜️"
    s_ok = "✅" if st.session_state.get("df_sleep") is not None else "⬜️"
    r_ok = "✅" if st.session_state.get("df_recovery") is not None else "⬜️"
    st.caption(f"{w_ok} Workouts • {s_ok} Sleep • {r_ok} Recovery")

with right:
    if all_loaded():
        st.session_state.show_uploads = False
        if st.button("Change files"):
            for k in ["workouts", "sleep", "recovery", "df_workouts", "df_sleep", "df_recovery"]:
                st.session_state.pop(k, None)
            st.session_state.show_uploads = True
            st.rerun()
    else:
        st.session_state.show_uploads = True

# uploaders (fully hidden once loaded)
if st.session_state.show_uploads:
    with st.expander("Upload panel", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            up_workouts = st.file_uploader(
                "Workouts: clean_strong_workouts.csv", type=["csv"], key="workouts"
            )
        with c2:
            up_sleep = st.file_uploader("Sleep: clean_sleep_data.csv", type=["csv"], key="sleep")
        with c3:
            up_recovery = st.file_uploader(
                "Recovery: clean_recovery_data.csv", type=["csv"], key="recovery"
            )

        # IMPORTANT: persist + normalize immediately when uploads exist
        if up_workouts is not None and st.session_state.get("df_workouts") is None:
            st.session_state.df_workouts = normalize_workouts(load_df_from_upload(up_workouts))

        if up_sleep is not None and st.session_state.get("df_sleep") is None:
            st.session_state.df_sleep = normalize_sleep(load_df_from_upload(up_sleep))

        if up_recovery is not None and st.session_state.get("df_recovery") is None:
            st.session_state.df_recovery = normalize_recovery(load_df_from_upload(up_recovery))


# Stop until all 3 are loaded (AFTER the persist step above)
# ---------- downstream dataframes (always from session_state) ---------
if not all_loaded():
    st.info("Upload the 3 CSVs to unlock the dashboard.")
    st.stop()
# Use dataframes ONLY from session_state from here onward
workouts = st.session_state.df_workouts
sleep = st.session_state.df_sleep
recovery = st.session_state.df_recovery

# -------------------------
# Tabs
# -------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🏠 Home",
        "🏋️ Workouts",
        "😴 Sleep",
        "🧠 Recovery",
        "🔗 Time Series Analysis",
        "📉 Stats",
        "⚙️ Models",
    ]
)
# =========================
# TAB 0 — HOME
# =========================

with tab0:
    sleep = home_tab.render(workouts, sleep, recovery)
# =========================
# TAB 1 — WORKOUTS
# =========================
with tab1:
    workouts_tab.render(workouts, cva_dt, smooth_days)

# =========================
# TAB 2 — SLEEP
# =========================
with tab2:
    sleep = sleep_tab.render(sleep)


# =========================
# TAB 3 — RECOVERY
# =========================
with tab3:
    recovery = recovery_tab.render(recovery)

# =========================
# TAB 4 — Time Series Analysis
# =========================
with tab4:
    time_series_analysis = timeseries_tab.render(recovery)


# =========================
# TAB 5 — STATS
# =========================
with tab5:
    stats_tab.render(recovery, workouts)
# =========================
# TAB 6 — MODELS
# =========================
with tab6:
    models_tab.render(time_series_analysis)

st.caption(
    "Tip: If you only train 3–4 days/week, use weekly aggregation (Volume / mean Recovery / mean Sleep) "
    "to avoid the mismatch between daily sleep and training frequency."
)
