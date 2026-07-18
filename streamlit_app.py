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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, classification_report, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve
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

st.set_page_config(page_title="Rehab Strength APP", layout="wide")
st.title("🏋️‍♂️ Rehab Strength APP", text_alignment="center")
st.caption("Workouts (Strong) • Sleep (Sheets) • Recovery (Sigmoid)")
app_version = "V2.3.4"
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
        unsafe_allow_html=True
    )
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "savefig.facecolor": "#0e1117",
        "text.color": "#e5e7eb",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#e5e7eb",
        "ytick.color": "#e5e7eb",
        "axes.edgecolor": "#e5e7eb",
        "grid.color": "#2d3748"
    })
else:
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

# -------------------------
# Helpers
# -------------------------

def make_unique_columns(cols):
    """Fix duplicate column names by suffixing .1 .2 ... (Streamlit upload sometimes preserves dupes)."""
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out
def pick_col(df, candidates):
    """Return the first column in candidates that exists in df, else None."""
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_date(df, col="Date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("D")
    return df

def epley_1rm(w, r):
    try:
        w = float(w); r = float(r)
        if np.isnan(w) or np.isnan(r):
            return np.nan
        return w * (1.0 + r/30.0)
    except Exception:
        return np.nan

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def daily_ma(series, window_days):
    # series is indexed by datetime daily (or resampled)
    return series.rolling(window=window_days, min_periods=max(1, window_days//2)).mean()

def weekly_bucket(dt_series):
    dt = pd.to_datetime(dt_series, errors="coerce")  # coerce bad strings to NaT
    return dt.dt.to_period("W-MON").dt.start_time

def plot_line(dfx, x, y, title, ylabel, xlabel="Date", marker="o", markersize=4, color=None,
              show_grid=True, despine=True, rotate_x=False, date_locator=None,
              date_formatter=None, linewidth=1.5):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfx[x], dfx[y], marker=marker, markersize=markersize, color=color, linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_grid:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    if rotate_x:
        ax.tick_params(axis='x', rotation=45)

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

def week_bounds(today=None):
    """Monday -> Sunday"""
    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()
    start = today - pd.Timedelta(days=today.weekday())
    end = start + pd.Timedelta(days=6)
    return start, end

def safe_minimal_last(df, date_col, value_col):
    if df is None or value_col is None:
        return None
    if date_col not in df.columns or value_col not in df.columns:
        return None
    tmp = df[[date_col, value_col]].copy()
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if tmp.empty:
        return None
    return tmp[value_col].iloc[-1]
def recovery_zone(x):
    if x is None or pd.isna(x):
        return "No data"
    if x >= 0.7: return "🟢 ⬆️ Ready"
    if x >= 0.55: return "🟡 Moderate"
    return "🔴 ⬇️ Low"

def correlation_insight(df, col1, col2):
    """Provide insight on correlation between two columns."""
    if df is None or col1 not in df.columns or col2 not in df.columns:
        return "Insufficient data for correlation analysis."
    corr_coef = df[[col1, col2]].dropna().corr().iloc[0, 1]
    if corr_coef == 1:
        return st.success(f"Perfect positive correlation (1.00) between {col1} and {col2}.")
    elif corr_coef > 0.7:
        return st.success(f"Strong positive correlation ({corr_coef:.2f}) between {col1} and {col2}.")
    elif corr_coef > 0.49:
        return st.info(f"Moderate positive correlation ({corr_coef:.2f}) between {col1} and {col2}.")
    elif corr_coef > 0:
        return st.warning(f"Weak or no significant correlation ({corr_coef:.2f}) between {col1} and {col2}.")
    elif corr_coef == -1:
        return st.success(f"Perfect negative correlation (1.00) between {col1} and {col2}.")
    elif corr_coef < -0.7:
        return st.success(f"Strong negative correlation ({corr_coef:.2f}) between {col1} and {col2}.")
    elif corr_coef < -0.49:
        return st.info(f"Moderate negative correlation ({corr_coef:.2f}) between {col1} and {col2}.")
    else:
        return st.warning(f"Weak or no significant correlation ({corr_coef:.2f}) between {col1} and {col2}.")

def sleep_classifier(q):
    return 1 if q in ["Good", "Excellent"] else 0

def string_to_decimal_hours(time_str):
    if pd.isna(time_str):
        return np.nan
    time_str = time_str.strip()
    if 'h' in time_str and 'min' in time_str:
        hours, minutes = time_str.split('h')
        minutes = minutes.replace('min', '').strip()
        return float(hours.strip()) + float(minutes) / 60
    elif 'h' in time_str:
        hours = time_str.replace('h', '').strip()
        return float(hours)
    elif 'min' in time_str:
        minutes = time_str.replace('min', '').strip()
        return float(minutes) / 60
    else:
        return np.nan

# -------------------------
# Uploads (hidden after loaded)
# -------------------------

# ---------- helpers ----------
def all_loaded() -> bool:
    return all(st.session_state.get(k) is not None for k in ["df_workouts", "df_sleep", "df_recovery"])

def load_df_from_upload(uploaded_file):
    data = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(data))

def normalize_workouts(df):
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    if "EXERCISE_NAME" in df.columns:
        df["EXERCISE_NAME"] = df["EXERCISE_NAME"].astype(str).str.strip()
    for col in ["WEIGHT_LBS", "REPS", "RPE", "VOLUME"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if set(["WEIGHT_LBS", "REPS"]).issubset(df.columns):
        df["est_1RM"] = df.apply(lambda r: epley_1rm(r["WEIGHT_LBS"], r["REPS"]), axis=1)

    if "DATE" in df.columns:
        df["Date"] = df["DATE"].dt.floor("D")
        df["DAY"] = df["Date"]
    return df

def normalize_sleep(df):
    df.columns = make_unique_columns(df.columns)
    if "Date" not in df.columns:
        for cand in ["DATE", "day", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    df = coerce_date(df, "Date")
    for cand in ["Score", "Wake Count", "Efficiency", "Asleep hrs", "InBed hrs",
                 "REM hrs", "Light hrs", "Deep hrs"]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df

def normalize_recovery(df):
    df.columns = make_unique_columns(df.columns)
    if "Date" not in df.columns:
        for cand in ["DATE", "day", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    df = coerce_date(df, "Date")
    for cand in ["Sigmoid Recovery Score", "RECOVERY_SCORE_RAW", "Stress_prev_day",
                 "Overnight HRV", "Resting Heart Rate", "Score"]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df

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
            up_workouts = st.file_uploader("Workouts: clean_strong_workouts.csv", type=["csv"], key="workouts")
        with c2:
            up_sleep = st.file_uploader("Sleep: clean_sleep_data.csv", type=["csv"], key="sleep")
        with c3:
            up_recovery = st.file_uploader("Recovery: clean_recovery_data.csv", type=["csv"], key="recovery")

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
sleep    = st.session_state.df_sleep
recovery = st.session_state.df_recovery

# -------------------------
# Tabs
# -------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Home", "🏋️ Workouts", "😴 Sleep", "🧠 Recovery", "🔗 Time Series Analysis", "📉 Stats", "⚙️ Models"])
# =========================
# TAB 0 — HOME
# =========================

with tab0:
    st.header("🏠 Weekly Snapshot")
    start_wk, end_wk = week_bounds()
    st.caption(f"Week: {start_wk.date()} → {end_wk.date()}")
    st.markdown("---")
    # compute last dates safely
    st.subheader("🔐 Data Integrity")
    last_workouts = workouts["Date"].max() if "Date" in workouts.columns else workouts["DATE"].max()
    last_sleep    = sleep["Date"].max() if sleep is not None and "Date" in sleep.columns else None
    last_recovery = recovery["Date"].max() if recovery is not None and "Date" in recovery.columns else None

    # freshness (days old)
    today = pd.Timestamp.today().normalize()
    def age_days(ts):
        if ts is None or pd.isna(ts):
            return None
        return int((today - pd.to_datetime(ts).normalize()).days)

    a_w = age_days(last_workouts)
    a_s = age_days(last_sleep)
    a_r = age_days(last_recovery)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.4])
    c1.metric("Workouts last", f"{pd.to_datetime(last_workouts):%b %d}", f"{a_w}d old" if a_w is not None else "—",
              delta_arrow="off")
    c2.metric("Sleep last",    f"{pd.to_datetime(last_sleep):%b %d}" if last_sleep is not None else "—",
            f"{a_s}d old" if a_s is not None else "—", delta_arrow="off")
    c3.metric("Recovery last", f"{pd.to_datetime(last_recovery):%b %d}" if last_recovery is not None else "—",
            f"{a_r}d old" if a_r is not None else "—", delta_arrow="off")

    # optional: overall status label
    overall_age = max([x for x in [a_w, a_s, a_r] if x is not None], default=None)
    label = "🟢 Fresh" if (overall_age is not None and overall_age <= 1) else ("🟡 Slightly delayed" if (overall_age is not None and overall_age <= 3) else "🔴 Outdated")
    c4.markdown(f"**Data status:** {label}")
    st.markdown("---")
     # -------------------------
    # Weekly workouts snapshot
    # -------------------------
    if workouts is None:
        st.info("Upload workouts CSV to see weekly snapshot.")
    else:
        if "DAY" not in workouts.columns or workouts["DAY"].isna().all():
            st.warning("Workouts CSV needs a DATE/Date column so the app can create 'DAY'.")
        else:
            wk = workouts[(workouts["DAY"] >= start_wk) & (workouts["DAY"] <= end_wk)].copy()

            # Workouts (sessions): unique (DAY, WORKOUT_NAME) if available
            if "WORKOUT_NAME" in wk.columns:
                workouts_count = wk.dropna(subset=["WORKOUT_NAME"]).groupby(["DAY", "WORKOUT_NAME"]).ngroups
            else:
                workouts_count = wk["DAY"].nunique()

            # Time exercised: MAX duration per session, then sum
            total_minutes = None
            if "DURATION_MIN" in wk.columns:
                wk["DURATION_MIN"] = pd.to_numeric(wk["DURATION_MIN"], errors="coerce")

                if "WORKOUT_NAME" in wk.columns:
                    total_minutes = (
                        wk.dropna(subset=["DURATION_MIN"])
                          .groupby(["DAY", "WORKOUT_NAME"])["DURATION_MIN"]
                          .max()
                          .sum()
                    )
                else:
                    total_minutes = wk.groupby("DAY")["DURATION_MIN"].max().sum()

            total_hours = None if total_minutes is None else float(total_minutes) / 60.0

            # -------------------------
            # Latest values (sleep/recovery)
            # -------------------------
            st.subheader("📊 Key Metrics")
            last_sigmoid = safe_minimal_last(recovery, "Date", "Sigmoid Recovery Score") if recovery is not None else None
            sleep_score_col = pick_col(recovery, ["Score", "Sleep Score", "SleepScore", "SCORE"])
            sleep_hrv_col   = pick_col(recovery, ["Overnight HRV", "Avg. HRV", "HRV", "7d Avg"])

            last_sleep_score = safe_minimal_last(recovery, "Date", sleep_score_col)
            last_hrv        = safe_minimal_last(recovery, "Date", sleep_hrv_col)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Workouts (week)", int(workouts_count) if workouts_count is not None else "—")
            c2.metric("Time exercised (hrs)", f"{total_hours:.1f}" if total_hours is not None else "—",
                      delta=f"4 Hrs goal", delta_arrow="off",
                      delta_color="normal" if total_hours is not None and total_hours >=4 else "inverse")
            if last_sigmoid is None or pd.isna(last_sigmoid):
                c3.metric("Last Recovery", "—", "No data")
            else:
                c3.metric(
                    "Last Recovery",
                    f"{last_sigmoid:.3f}",
                    recovery_zone(last_sigmoid), delta_arrow="off",
                    delta_color="normal" if last_sigmoid is not None and last_sigmoid >= 0.7 else ("inverse" if last_sigmoid is not None and last_sigmoid >= 0.55 else "inverse"))
            c4.metric("Last sleep score %", f"{float(last_sleep_score):.0f}" if last_sleep_score is not None else "—", f"Excellent" if last_sleep_score is not None and last_sleep_score >= 85 else ("Fair" if last_sleep_score is not None and last_sleep_score >= 70 else "Poor"), delta_arrow="off")
            c5.metric("Last HRV (ms)", f"{float(last_hrv):.0f}" if last_hrv is not None and str(last_hrv) != "nan" else "—",
                      delta="Bad" if last_hrv is not None and last_hrv < 45 else "Good" if last_hrv is not None and last_hrv <= 60 else "Excellent",
                      delta_arrow="off", delta_color="normal" if last_hrv is not None and last_hrv >= 45 else ("inverse" if last_hrv is not None and last_hrv < 60 else "off"))

            st.subheader("📈 Recent Trends")
            last_sigmoid_nap = safe_minimal_last(recovery, "Date", "Sigmoid with Nap") if recovery is not None else None
            last_delta = safe_minimal_last(recovery, "Date", "DELTA_NAP") if recovery is not None else None
            last_nap_status = safe_minimal_last(recovery, "Date", "Nap_Status") if recovery is not None else None
            c1, c2, c3 = st.columns(3)
            c1.metric("Recovery with Nap", f"{last_sigmoid_nap:.3f}" if last_sigmoid_nap is not None else "—",
                      recovery_zone(last_sigmoid_nap), delta_arrow="off",
                      delta_color="normal" if last_sigmoid_nap is not None and last_sigmoid_nap > last_sigmoid else ("inverse" if last_sigmoid_nap is not None and last_sigmoid_nap < last_sigmoid else "off"))
            c2.metric("Δ Nap Effect", f"{last_delta:.3f}" if last_delta is not None else "—",
                      f"{'⬆️ Positive' if last_delta is not None and last_delta > 0 else ('⬇️ Negative' if last_delta is not None and last_delta < 0 else 'No effect')}",
                      delta_arrow="off", delta_color="normal" if last_delta is not None and last_delta > 0 else ("inverse" if last_delta is not None and last_delta < 0 else "off"))
            c3.metric("Last Nap Status", last_nap_status if last_nap_status is not None else "No nap", delta_arrow="off")
    st.markdown("---")

    #---------------------------
    # Naps summary
    #--------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("💤 Naps logged")
        window = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window",
            selection_mode="single", format_func=lambda x: f"{x} days")
        if window is None:
            window = 14
            st.markdown("Please select a number of days")
        sleep = sleep.sort_values("Date", ascending=True)
        avg_nap = sleep.tail(window)["Asleep_Nap"].dropna().mean()
        try:
            if pd.isna(avg_nap):
                col1.metric(f"💤 Avg nap time (last {window} days)", " 0 minutes")
                st.write("No Naps Logged")
            else:
                col1.metric(f"💤 Avg nap time (last {window} days)", f"{avg_nap:.1f} minutes")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")

    with col2:
        st.subheader("📅 Nap Days")
        window2 = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window2",
            selection_mode="single", format_func=lambda x: f"{x} days")
        try:
            if pd.isna(avg_nap):
                st.write("No Day Naps Logged")
            else:
                start_date = sleep["Date"].max() - pd.Timedelta(days=window2)
                nap_data = sleep[sleep["Date"] >= start_date].dropna(subset=["Asleep_Nap"])
                n_naps = nap_data.loc[nap_data["Asleep_Nap"].notna() & (nap_data["Asleep_Nap"] > 0)].shape[0]
                col2.metric("Naps logged", f"{n_naps}")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")
    with col3:
        st.subheader("📈 Nap Frequency")
        window3 = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window3",
            selection_mode="single", format_func=lambda x: f"{x} days")
        try:
            if pd.isna(avg_nap):
                st.write("No Frequency Naps Logged")
            else:
                start_date = sleep["Date"].max() - pd.Timedelta(days=window3)
                nap_data = sleep[sleep["Date"] >= start_date].dropna(subset=["Asleep_Nap"])
                sleep_filtered = sleep[sleep["Date"] >= start_date]
                freq_val = (nap_data[nap_data["Asleep_Nap"].notna() & (nap_data["Asleep_Nap"] > 0)].shape[0] / sleep_filtered.shape[0]) * 100
                def freq_label(x):
                    if pd.isna(x):
                        return "No data"
                    if x <= 15: return "🔴 Low"
                    if x <= 30: return "🟡 Moderate"
                    return "🟢 High"
                col3.metric("Nap frequency", f"{freq_val:.1f} %", delta=freq_label(freq_val))
                st.caption(f"Naps on {nap_data[nap_data['Asleep_Nap'].notna() & (nap_data['Asleep_Nap'] > 0)].shape[0]} of the last {window3} days")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")
    st.markdown("---")
    # -------------------------
    # Quick trends (independent)
    # -------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("🧠 Recovery (last 14 days)")
        if recovery is not None and {"Date", "Sigmoid Recovery Score"}.issubset(recovery.columns):
            tmp = recovery.dropna(subset=["Date", "Sigmoid Recovery Score"]).sort_values("Date").tail(14)
            if tmp.empty:
                st.info("Recovery CSV loaded but no usable rows.")
            else:
                fig, ax = plt.subplots(figsize=(7,3))
                tmp_avg = tmp["Sigmoid Recovery Score"].mean()
                sld1 = st.slider("Select days for moving average", 2, 11, 5, 1, width=250)
                ax.plot(tmp["Date"], tmp["Sigmoid Recovery Score"], marker="o", markersize=3, color="green")
                roll_avg_recovery = tmp["Sigmoid Recovery Score"].rolling(window=sld1).mean()
                ax.plot(tmp["Date"], roll_avg_recovery, color="orange", linestyle="--",
                        alpha=0.25, label=f"MA {sld1} days {roll_avg_recovery.iloc[-1]:.2f}")
                ax.axhline(tmp_avg, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp_avg:.2f}")
                ax.set_xlabel("")
                ax.set_ylim(0, 1)
                ax.legend(loc="lower left")
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_ylabel("Sigmoid")
                sns.despine(ax=ax)
                st.pyplot(fig)
        else:
            st.info("No recovery data uploaded yet.")

    with right:
        st.subheader("🧠 Recovery (last 14 days) with Nap")
        if recovery is not None and {"Date", "Sigmoid with Nap"}.issubset(recovery.columns):
            tmp2 = recovery.dropna(subset=["Date", "Sigmoid with Nap"]).sort_values("Date").tail(14)
            last_recovery_nap = tmp2["Sigmoid with Nap"].iloc[-1] if not tmp2.empty else None
            if tmp2.empty:
                st.info("Recovery CSV loaded but no usable rows.")
            else:
                fig, ax = plt.subplots(figsize=(7,3))
                tmp2_avg = tmp2["Sigmoid with Nap"].mean()
                sld_sig_nap = st.slider("Select days for moving average", 2, 11, 5, 1, width=250, key="sig_nap_ma_slider")
                ax.plot(tmp2["Date"], tmp2["Sigmoid with Nap"], marker="o", markersize=3, color="seagreen")
                roll_avg_recovery_nap = tmp2["Sigmoid with Nap"].rolling(window=sld_sig_nap).mean()
                ax.plot(tmp2["Date"], roll_avg_recovery_nap, color="orange", linestyle="--",
                        alpha=0.25, label=f"MA {sld_sig_nap} days {roll_avg_recovery_nap.iloc[-1]:.2f}")
                ax.axhline(tmp2_avg, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp2_avg:.2f}")
                ax.set_xlabel("")
                ax.set_ylim(0,1)
                ax.legend(loc="lower left")
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_ylabel("Sigmoid")
                sns.despine(ax=ax)
                st.pyplot(fig)
        else:
            st.info("No recovery data uploaded yet.")
    with st.expander("Recovery Insights", icon="🧠",expanded=False):
        vol_no_nap = recovery["Sigmoid Recovery Score"].tail(14).std()
        vol_nap = recovery["Sigmoid with Nap"].tail(14).std()
        delta_window = st.slider("Select days window for Δ average", 2, 14, 7, 1, key="delta_avg_window", width=200)
        recovery_naps = recovery.dropna(subset=["Date", "DELTA_NAP"]).sort_values("Date").tail(delta_window)
        avg_delta = recovery_naps["DELTA_NAP"].mean() if not recovery_naps.empty else None
        st.write(f"Average Δ {delta_window} days: {avg_delta:.2f}" if avg_delta is not None else "Average Δ not available")
        if last_sigmoid is not None and last_sigmoid >= tmp_avg:
            st.write(f"14-day Last Recovery without Nap: {last_sigmoid:.2f} above Avg")
        else:
            st.write(f"14-day Last Recovery without Nap: {last_sigmoid_nap:.2f} below Avg")
        if last_recovery_nap is not None and last_recovery_nap >=tmp2_avg:
            st.write(f"14-day Last Recovery with Nap: {last_recovery_nap:.2f} above Avg")
        else:
            st.write(f"14-day Last Recovery with Nap: {last_recovery_nap:.2f} below Avg")
        st.write(f"Volatility (STD) without Nap (14 days): {vol_no_nap:.4f}")
        st.write(f"Volatility (STD) with Nap (14 days): {vol_nap:.4f}")

    st.caption("Note: Recovery with Nap may be higher than without nap depending on nap effect.", help="Nap effect is computed based on the duration of the nap and the hour it was taken. A positive nap effect indicates that the nap contributed positively to recovery, while a negative effect suggests it may have disrupted sleep patterns.")
    st.markdown("---")
    st.subheader("😴 Sleep score (last 14 days)")
    sleep_score_col = pick_col(recovery, ["Score", "Sleep Score", "SleepScore", "SCORE", "Score.1", "Score.2"]) if recovery is not None else None

    if recovery is not None and sleep_score_col is not None and "Date" in recovery.columns:
        tmp = recovery.dropna(subset=["Date", sleep_score_col]).sort_values("Date").tail(14)
        if tmp.empty:
            st.info("Sleep CSV loaded but no usable rows.")
        else:
            fig, ax = plt.subplots(figsize=(7,3))
            sld2 = st.slider("Select days for moving average", 2, 11, 5, 1, key="sleep_ma_slider", width=250)
            roll_avg_sleep = tmp[sleep_score_col].rolling(window=sld2).mean()
            tmp_avg_sleep = tmp[sleep_score_col].mean()
            ax.axhline(tmp_avg_sleep, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp_avg_sleep:.0f}")
            ax.plot(tmp["Date"], tmp[sleep_score_col], marker="o", markersize=3, color="purple")
            ax.plot(tmp["Date"],roll_avg_sleep, color="orange", linestyle="--",
                    alpha=0.4, label=f"MA {sld2} days {roll_avg_sleep.iloc[-1]:.0f}")
            ax.set_xlabel("")
            ax.legend(loc="lower left")
            ax.set_ylim(50,100)
            ax.set_ylabel(sleep_score_col)
            ax.tick_params(axis='x', rotation=45, labelsize=6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            sns.despine(ax=ax)
            st.pyplot(fig)
    else:
        st.info("No sleep data uploaded yet (or score column not detected).")
    st.markdown("---")

    with st.expander("🛠 Debug"):
        st.write("Workouts cols:", list(workouts.columns))
        st.write("Sleep cols:", list(sleep.columns))
        st.write("Recovery cols:", list(recovery.columns))
    st.markdown("--")
# =========================
# TAB 1 — WORKOUTS
# =========================
with tab1:
    st.header("🏋️ Workouts")

    if workouts is None:
        st.info("Upload your cleaned workouts CSV to see charts.")
    else:
        # basic checks
        req = {"Date", "EXERCISE_NAME"}
        if not req.issubset(workouts.columns):
            st.error(f"Workouts CSV must include at least: {req}")
        else:
            cva_ts = pd.to_datetime(cva_dt)

            # pick exercise
            ex_list = sorted(workouts["EXERCISE_NAME"].dropna().unique())
            chosen_ex = st.selectbox("Choose an exercise:", ex_list)

            w = workouts[workouts["EXERCISE_NAME"] == chosen_ex].copy()
            w = w.dropna(subset=["Date"]).sort_values("Date")

            # -------- 1) Pre vs Post (Estimated 1RM mean)
            st.subheader("📊 Pre vs Post (Estimated 1RM)")
            if "est_1RM" in w.columns:
                pre = w[w["Date"] < cva_ts]["est_1RM"].mean()
                post = w[w["Date"] >= cva_ts]["est_1RM"].mean()

                fig, ax = plt.subplots(figsize=(6, 4))
                vals = [pre, post]
                labs = ["Pre-CVA", "Post-CVA"]
                ax.bar(np.arange(2), vals, width=0.6, edgecolor="black")
                ax.set_xticks(np.arange(2)); ax.set_xticklabels(labs)
                ax.set_ylabel("Estimated 1RM (lb)")
                ax.set_title(chosen_ex, fontsize=14, fontweight="bold", pad=15)
                sns.despine(ax=ax)
                for i, v in enumerate(vals):
                    if not np.isnan(v):
                        ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom")
                ax.grid(True, axis="y", alpha=0.25)
                ax.set_axisbelow(True)
                st.pyplot(fig)
            else:
                st.info("No est_1RM found. Make sure WEIGHT_LBS and REPS exist in the workouts file.")

            # -------- 2) Progress over time (daily + MA)
            st.subheader("⏳ Progress over time (Daily + Moving Avg)")
            if "est_1RM" in w.columns:
                # daily mean est_1RM
                daily = w.groupby("Date", as_index=False)["est_1RM"].mean().sort_values("Date")
                daily["MA"] = daily_ma(daily["est_1RM"], smooth_days)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(daily["Date"], daily["est_1RM"], marker="s", label="Daily mean est.1RM", color="salmon", markersize=4)
                ax.plot(daily["Date"], daily["MA"], linestyle="--", label=f"{smooth_days}-day MA", color="yellow")
                ax.axvline(cva_ts, linestyle=":", linewidth=1)
                ax.set_title(f"{chosen_ex} — Comparative Pre & Post CVA", fontsize=14, fontweight="bold", pad=15)
                ax.set_xlabel("Date"); ax.set_ylabel("lb")
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
                sns.despine(ax=ax)
                ax.legend()
                st.pyplot(fig)

            # -------- 3) Weekly Volume chart (per exercise + total)
            st.subheader("📦 Weekly Volume (Exercise and Total)")
            if "VOLUME" in workouts.columns:
                workouts["Week"] = weekly_bucket(workouts["Date"])
                w_ex_week = workouts[workouts["EXERCISE_NAME"] == chosen_ex].groupby("Week", as_index=False)["VOLUME"].sum()
                w_all_week = workouts.groupby("Week", as_index=False)["VOLUME"].sum()

                cA, cB = st.columns(2)
                with cA:
                    plot_line(w_ex_week.sort_values("Week"), "Week", "VOLUME",
                              f"Weekly Volume — {chosen_ex}", "Total Volume (lb·reps)", xlabel="Week")
                with cB:
                    plot_line(w_all_week.sort_values("Week"), "Week", "VOLUME",
                              "Weekly Volume — ALL Exercises", "Total Volume (lb·reps)", xlabel="Week")

            # -------- 4) RPE trend (daily mean)
            st.subheader("🔥 RPE Trend (Daily)")
            if "RPE" in w.columns:
                rpe_daily = w.groupby("Date", as_index=False)["RPE"].mean().sort_values("Date")
                if rpe_daily["RPE"].notna().sum() == 0:
                    st.info("No RPE values recorded for this exercise yet.")
                else:
                    rpe_daily["MA"] = daily_ma(rpe_daily["RPE"], smooth_days)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(rpe_daily["Date"], rpe_daily["RPE"], marker="o", label="Daily mean RPE", color="salmon", markersize=4)
                    ax.plot(rpe_daily["Date"], rpe_daily["MA"], linestyle="--", label=f"{smooth_days}-day MA", color="yellow")
                    ax.axvline(cva_ts, linestyle=":", linewidth=1)
                    ax.set_title(f"{chosen_ex} — RPE Trend & Post CVA", fontsize=14, fontweight="bold", pad=15)
                    ax.set_xlabel("Date"); ax.set_ylabel("RPE")
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                    ax.tick_params(axis='x', rotation=45)
                    sns.despine(ax=ax)
                    ax.legend()
                    st.pyplot(fig)

            # -------- Summary table (exercise)
            with st.expander("📋 Show raw sets for this exercise"):
                st.dataframe(w.sort_values("DATE" if "DATE" in w.columns else "Date"))

# =========================
# TAB 2 — SLEEP
# =========================
with tab2:
    st.header("😴 Sleep")

    if sleep is None:
        st.info("Upload your clean sleep CSV to see charts.")
    else:
        if "Date" not in sleep.columns:
            st.error("Sleep CSV must include a Date column.")
        else:
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
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
                ax.set_axisbelow(True)
                ax.legend()
                st.pyplot(fig)

            # Wake Count
            if "Wake Count" in sleep.columns:
                st.subheader("🌙 Wake Count")
                plot_line(sleep.dropna(subset=["Wake Count"]), "Date", "Wake Count",
                        "Wake Count over time", "Count",
                        marker=None, color="purple", xlabel="",
                        rotate_x=True, date_locator=mdates.MonthLocator(interval=2), linewidth=0.7)
            # Naps
            left, right = st.columns(2)
            if "Asleep_Nap" in sleep.columns:
                with left:
                    st.subheader("💤 Nap Asleep (min)")
                    slider_nap2 = st.slider("Select number of days to show for Nap Asleep plot", 60, 365, 365, 1, key="nap_asleep_days")
                    recent_date = sleep["Date"].max()
                    start_date = recent_date - pd.Timedelta(days=slider_nap2)
                    filtered_nap = sleep[(sleep["Date"] >= start_date) & (sleep["Date"] <= recent_date)].copy()
                    df_plot = filtered_nap.dropna(subset=["Asleep_Nap"])
                    roll_avg_nap = df_plot["Asleep_Nap"].rolling(window=7, min_periods=1).mean()


                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_plot["Date"], df_plot["Asleep_Nap"], color="teal", linewidth=1.5, label="Nap Asleep")
                    ax.plot(df_plot["Date"], roll_avg_nap, marker="o", markersize=2, color="salmon", label=f"7-day MA",
                            linewidth=1, linestyle="--")
                    ax.set_title("Nap Asleep over time")
                    ax.set_ylabel("Minutes")
                    ax.tick_params(axis='x', rotation=45)
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
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
                    slider_nap = st.slider("Select number of days to show for Nap Asleep plot", 60, 365, 365, 1, key="nap_asleep")
                    recent_date = sleep["Date"].max()
                    start_date = recent_date - pd.Timedelta(days=slider_nap)
                    sleep_filtered = sleep[(sleep["Date"] >= start_date) & (sleep["Date"] <= recent_date)].copy()
                    df_nap = sleep_filtered.dropna(subset=["Asleep_Nap"])[["Date", "Asleep_Nap"]].copy()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(df_nap["Date"], df_nap["Asleep_Nap"], color="teal", width=0.8)
                    ax.set_title("Nap Asleep over time")
                    ax.set_ylabel("Minutes")
                    ax.tick_params(axis='x', rotation=45)
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                    sns.despine(ax=ax)
                    st.pyplot(fig)

                    # Monthly total nap inbed
                    sleep_monthly = sleep_filtered.set_index("Date").resample("M")["Asleep_Nap"].sum().reset_index()
                    st.subheader("🗓️ Monthly Nap Asleep Total (min)")
                    plot_line(sleep_monthly.dropna(subset=["Asleep_Nap"]), "Date", "Asleep_Nap",
                            "Monthly Nap Asleep Total", "Minutes",
                            marker="o", color="coral", xlabel="",
                            rotate_x=True, date_locator=mdates.MonthLocator(interval=1), show_grid=True, date_formatter=mdates.DateFormatter('%b-%Y'))
                else:
                    st.info("Column 'InBed_Nap' not found in sleep data.")


# =========================
# TAB 3 — RECOVERY
# =========================
with tab3:
    st.header("🧠 Recovery")

    if recovery is None:
        st.info("Upload your clean recovery CSV to see charts.")
    else:
        if "Date" not in recovery.columns:
            st.error("Recovery CSV must include a Date column.")
        else:
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
                    "Score", xlabel="", color="seagreen", rotate_x=True,
                    date_locator=mdates.DayLocator(interval=2),
                    date_formatter=mdates.DateFormatter('%b-%d')
                )

            # Components (choose what you want)
            st.subheader("🧩 Components")
            candidates = [
                "Stress_prev_day",
                "Overnight HRV",
                "Resting Heart Rate",
                "Score",
                "RECOVERY_SCORE_RAW",
            ]
            available = [c for c in candidates if c in recovery.columns]
            if available:
                chosen = st.multiselect("Pick component(s) to plot:", available, default=available[:3])
                for col in chosen:
                    plot_line(recovery.dropna(subset=[col]), "Date", col, f"{col} over time", col)
            else:
                st.info("No component columns detected (Stress_prev_day / Overnight HRV / etc.).")

# =========================
# TAB 4 — Time Series Analysis
# =========================
with tab4:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller, kpss
    st.header("🔗 Time Series Analysis")

    tsa_col = ["Date", "Start", "End", "InBed hrs", "Asleep hrs", "Awake", "REM hrs", "Light hrs", "Deep hrs",
    "Efficiency", "Fall Asleep", "Score"]
    tsa_df = recovery[tsa_col].dropna(subset=["Score"]).copy().sort_values(by="Date", ascending=True) if recovery is not None else None
    time_series_analysis = ""
    tsa_series = (
        tsa_df.set_index("Date")["Score"]
        .asfreq("D")           # insert NaN for missing nights
        .interpolate("time")   # fill gaps proportionally to time distance
    )

    with st.expander("📋 Time Series Data", expanded=False):
        st.write("Length of time series data:", tsa_df.shape[0] if tsa_df is not None else "N/A")
        st.dataframe(tsa_df.head(10))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tsa_df["Date"], tsa_df["Score"], marker="x", markersize=2, color="salmon", linewidth=0.5)
        ax.set_title("Sleep Score over time", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
        st.pyplot(fig)

    with st.expander("📊 ACF & PACF of Sleep Score", expanded=True):
        max_lags = min(100, len(tsa_series) // 2 - 1)
        LAGS = st.slider("Select number of lags for ACF/PACF", 10, max_lags, min(30, max_lags), 5)
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(tsa_series, lags=LAGS, title="ACF of Sleep Score", ax=ax)
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(tsa_series, lags=LAGS, title="PACF of Sleep Score", ax=ax)
        st.pyplot(fig)

    with st.expander("📈 Stationary Tests of Sleep Score", expanded=True):
        # ADF Test — H₀: series has a unit root (NON-stationary)
        st.info("ADF Test — H₀: series has a unit root")
        adf_result = adfuller(tsa_series)
        st.write(f"ADF statistic: {adf_result[0]:.4f}")
        st.write(f"ADF p-value:   {adf_result[1]:.4f}")
        # p < 0.05 → reject H₀ → stationary ✅
        pvalue_adf = adf_result[1]
        # KPSS Test — H₀: series IS stationary (OPPOSITE null!)
        st.info("KPSS Test — H₀: series IS stationary")
        kpss_result = kpss(tsa_series, regression="ct", nlags="auto")
        st.write(f"KPSS statistic: {kpss_result[0]:.4f}")
        st.write(f"KPSS p-value:   {kpss_result[1]:.4f}")
        # p > 0.05 → fail to reject H₀ → stationary ✅
        pvalue_kpss = kpss_result[1]


        if pvalue_adf < 0.05 and pvalue_kpss > 0.05:
            st.success("Both tests indicate the series is likely stationary.")
            time_series_analysis = "Stationary"
        elif pvalue_adf >= 0.05 and pvalue_kpss <= 0.05:
            st.warning("Both tests indicate the series is likely non-stationary.")
            time_series_analysis = "Non-Stationary"
        else:
            st.info("Tests are inconclusive or conflicting. Consider differencing or further analysis.")
            time_series_analysis = "Inconclusive"
        st.write(f"Overall Time Series Analysis: **{time_series_analysis}**")


# =========================
# TAB 5 — STATS
# =========================
with tab5:
    # Data agg necessary for hypothesis testing
    recovery["Date"] = pd.to_datetime(recovery["Date"], errors="coerce")  # Convert to datetime
    workouts_daily = workouts.groupby("Date", as_index=False).agg({
        "DURATION_MIN": "max",
        "WEIGHT_LBS": "max",
        "VOLUME": "sum",
        "RPE": "mean",
        "est_1RM": "mean"})
    recovery_exercise = pd.merge(recovery, workouts_daily, on="Date", how="left").sort_values("Date")
    recovery_exercise["Exercise_Done"] = recovery_exercise["VOLUME"].fillna(0).gt(0).astype(int)
    recovery_exercise_done = recovery_exercise[recovery_exercise["Exercise_Done"] == 1].copy()
    recovery_exercise_notdone = recovery_exercise[recovery_exercise["Exercise_Done"] == 0].copy()
    #-----------------------------
    st.header("📉 Stats")
    st.subheader("📊 Recovery on Exercise vs Non-Exercise Days")
    allowed = sorted(["InBed hrs", "Asleep hrs", "Wake Count", "REM hrs", "Light hrs", "Deep hrs", "Fall Asleep",
            "Overnight HRV", "Stress", "RHR", "Score"])
    check_metric = st.selectbox("Select metric to analyze:", allowed, index=allowed.index("Score"))
    picked_col = recovery[check_metric] if recovery is not None and check_metric in recovery.columns else None
    mean_val = picked_col.mean()
    median_val = picked_col.median()
    std_val = picked_col.std()
    trim_mean_val = stats.trim_mean(picked_col.dropna(), 0.1) if picked_col is not None else None
    n = picked_col.dropna().shape[0]
    trim_mean = stats.trim_mean(picked_col.dropna(), 0.1) if picked_col is not None else None
    cv = std_val / mean_val if mean_val not in (0, None) and not pd.isna(mean_val) else np.nan
    col_pvalue, col_inter = normality_test(picked_col) if picked_col is not None else (None, None)
    #------------------------------------- 4 MOMENTS OF DATA ----------------------------------------
    with st.expander("🎯 Four Moments of Statistics", expanded=False):
        st.subheader("🎯 Four Moments of Statistics")
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        series_for_chart = picked_col.dropna().astype(float).tail(30).tolist() if picked_col is not None else []
        c1.metric("Median", f"{median_val:.2f}" if median_val is not None else "—")
        c2.metric(
            "Mean",
            f"{mean_val:.2f}" if mean_val is not None and not pd.isna(mean_val) else "—",
            chart_data=series_for_chart if len(series_for_chart) >= 2 else None,
            chart_type="line")
        c3.metric("Std Dev", f"{std_val:.2f}" if std_val is not None else "—")
        c4.metric("Trimmed Mean (10%)", f"{trim_mean:.2f}" if trim_mean is not None else "—")
        c5.metric("Sample (n)", "Sufficient" if n >= 30 else "Insufficient",
                  delta=n, delta_color="normal" if n >= 30 else "inverse",
                  help="n >=30 is considered sufficient for Central Limit Theorem.", delta_arrow="off")
        c6.metric("Coef of Var (CV)", "Good" if cv is not None and cv < 0.1 else "Acceptable" if cv is not None and cv < 0.2 else "High",
                  delta=f"{round(cv*100, 2)} %",delta_color="normal" if cv is not None and cv < 0.1 else "orange" if cv is not None and cv < 0.2 else "inverse" if cv is not None else None,
                  help="CV <10% is considered good stability; 10-20% acceptable; >20% high variability.")
        c7.metric("Skewness", f"{picked_col.skew():.2f}" if picked_col is not None else "—", help="Skewness indicates asymmetry. >0 means right-skewed, <0 means left-skewed.")
        c8.metric("Kurtosis", f"{picked_col.kurtosis():.2f}" if picked_col is not None else "—", help="Kurtosis indicates the 'tailedness' of the distribution. >3 means heavy tails, <3 means light tails.",
                  delta="Leptokurtic" if picked_col is not None and picked_col.kurtosis() > 3 else "Platykurtic" if picked_col is not None and picked_col.kurtosis() < 3 else "Mesokurtic" if picked_col is not None else None,
                  delta_arrow="off", delta_color="green" if picked_col is not None and picked_col.kurtosis() == 3 else "red" if picked_col is not None and picked_col.kurtosis() > 3 else "green" if picked_col is not None and picked_col.kurtosis() < 3 else None)
    #------------------------------------- EMPIRICAL CDF & PERCENTILES ----------------------------------------
    with st.expander("📊 Empirical CDF & Percentiles", expanded=False):
        st.subheader("📊 Empirical CDF & Percentiles")
        c1, c2 = st.columns(2)
        with c1:
            if picked_col is not None and not picked_col.dropna().empty:
                complementary = st.segmented_control("Complementary CDF ?:", [True, False], key="cdf_type_control", default=True,
                                                     help="If True then complementary CDF (1 - CDF) is shown.")
                perc_90 = picked_col.quantile(0.9)
                perc_75 = picked_col.quantile(0.75)
                fig, ax = plt.subplots(figsize=(10, 4))
                cecdf_50 = compute_ecdf(picked_col.dropna(), median_val, complementary=complementary)
                cecdf_75 = compute_ecdf(picked_col.dropna(), perc_75, complementary=complementary)
                cecdf_90 = compute_ecdf(picked_col.dropna(), perc_90, complementary=complementary)
                sns.ecdfplot(data=recovery, x=check_metric, label=f"Empirical CDF {check_metric}",
                            color="green", complementary=complementary, ax=ax)
                plt.axvline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}", linewidth=0.5)
                plt.axvline(median_val, color="lightseagreen", linestyle=":", label=f"50th Percentile: {median_val:.2f}", linewidth=1)
                plt.axvline(perc_90, color="yellow", linestyle="--", label=f"90th Percentile: {perc_90:.2f}", linewidth=0.5)
                plt.axvline(perc_75, color="brown", linestyle="--", label=f"75th Percentile: {perc_75:.2f}", linewidth=0.5)
                if complementary:
                    plt.title(f"Complementary ECDF of {check_metric}", fontsize=14, fontweight="bold", pad=15)
                    plt.ylabel("Complementary ECDF")
                else:
                    plt.title(f"Empirical CDF of {check_metric}", fontsize=14, fontweight="bold", pad=15)
                    plt.ylabel("ECDF")
                plt.xlabel(check_metric)
                plt.legend(loc="best", fontsize=7)
                sns.despine()
                st.pyplot(fig)
            else:
                st.warning(f"No data available for {check_metric} to plot ECDF.")
        with c2:
            st.subheader(f"📈 Percentile Insights {check_metric}")
            st.write("Typical value (median)", round(median_val,2))
            st.write("Uncommon value (75th ):", round(perc_75,2))
            st.write("Rare high value (90th):", round(perc_90,2))
            st.subheader(f"📊 Probability Insights {check_metric}")
            if complementary:
                st.write("The probability of exceeding", round(median_val,2), " is:" , round((cecdf_50)*100, 2), " %")
                st.write("The probability of exceeding", round(perc_75,2), " is:", round((cecdf_75)*100, 2), " %")
                st.write("The probability of exceeding", round(perc_90,2), " is:", round((cecdf_90)*100, 2), " %")
            else:
                st.write("The probability of getting any value up to my typical performance is:", round(cecdf_50*100, 2), " %")
                st.write("The probability of getting any value up to common performance is:", round(cecdf_75*100, 2), " %")
                st.write("The probability of getting any value up to atypical performance is:", round(cecdf_90*100, 2), " %")
            st.info("Note: Even though the ECDF provides empirical probabilities based on historical data, " \
            "it does not guarantee future outcomes. Use this information as a guide rather than a definitive prediction.",
            icon="ℹ️")
    #------------------------------------- NORMALITY TEST & VISUALS ----------------------------------------
    with st.expander("🔍 Normality Test for Recovery", expanded=False):
        st.subheader("🔍 Normality Test for Recovery")

        # Interpretation
        if col_pvalue is not None and col_pvalue > 0.05:
            st.success(f"Shapiro Wilk Test: {check_metric} appears to be normally distributed (p={col_pvalue:.3f}). You can use parametric tests.")
        elif col_pvalue is not None and col_pvalue <= 0.05:
            st.info(f"Shapiro Wilk Test: {check_metric} does not appear to be normally distributed (p={col_pvalue:.3f}). You may want to use non-parametric tests.")
        else:
            st.info(f"Not enough data to perform normality test on {check_metric}.")
        try:
            distributions = pd.to_numeric(picked_col, errors="coerce").dropna()
            st.dataframe(fit_distribution(distributions))
        except Exception as e:
            st.error(f"Error fitting distribution: {e}")
        bins = st.slider("Select number of bins for histogram", 5, 50, 20, 1, width=250, key="hist_bins_slider")
        c1, c2 = st.columns(2)
        with c1:
            #Plot histogram
            fig, ax = plt.subplots(figsize=(7,3))
            sns.histplot(picked_col.dropna(), kde=True, ax=ax, stat="probability", bins=bins)
            ax.axvline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="red", linestyle=":", label=f"Median: {median_val:.2f}")
            ax.axvline(trim_mean_val, color="green", linestyle="-.", label=f"Trimmed Mean: {trim_mean_val:.2f}")
            ax.axvspan(mean_val - std_val, mean_val + std_val, color="yellow", alpha=0.15, label=f"±1 Std Dev: {std_val:.2f}")
            ax.set_title(f"Histogram of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.legend(loc="best", fontsize=7)
            ax.set_xlabel(check_metric)
            ax.set_ylabel("Probability")
            sns.despine(ax=ax)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7,3))
            sns.boxplot(y=picked_col.dropna(), ax=ax, width=0.3, fliersize=3, flierprops={"markerfacecolor": "red", "marker": "o"})
            sns.stripplot(y=picked_col.dropna(), ax=ax, color="lightblue", size=4, jitter=True, alpha=0.15)
            ax.set_title(f"Boxplot of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel(check_metric)
            sns.despine(ax=ax)
            st.pyplot(fig)

        if  st.checkbox("Show full time series plot", value=True, key="full_time_series_checkbox"):
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.lineplot(data=recovery, x="Date", y=check_metric, ax=ax, linewidth=1)
            ax.axhline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.axhline(median_val, color="red", linestyle=":", label=f"Median: {median_val:.2f}")
            ax.axhline(trim_mean_val, color="green", linestyle="-.", label=f"Trimmed Mean: {trim_mean_val:.2f}")
            ax.axhspan(mean_val - std_val, mean_val + std_val, color="yellow", alpha=0.05, label=f"±1 Std Dev: {std_val:.2f}")
            ax.legend(loc="best", fontsize=5)
            ax.set_title(f"Time Series of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("")
            ax.set_ylabel(check_metric)
            sns.despine(ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        else:
            date_plot = st.slider("Select number of days to show for time series plot", 30, 365, 180, 1, key="time_series_days_slider",
                width=250)
            date_filter = recovery["Date"].max() - pd.Timedelta(days=date_plot)
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.lineplot(data=recovery[recovery["Date"] >= date_filter], x="Date", y=check_metric, ax=ax, linewidth=1)
            ax.axhline(mean_val, color="blue", linestyle="--", label=f"Mean: {mean_val:.2f}")
            ax.axhline(median_val, color="red", linestyle=":", label=f"Median: {median_val:.2f}")
            ax.axhline(trim_mean_val, color="green", linestyle="-.", label=f"Trimmed Mean: {trim_mean_val:.2f}")
            ax.axhspan(mean_val - std_val, mean_val + std_val, color="yellow", alpha=0.05, label=f"±1 Std Dev: {std_val:.2f}")
            ax.legend(loc="best", fontsize=5)
            ax.set_title(f"Time Series of {check_metric}", fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("")
            ax.set_ylabel(check_metric)
            sns.despine(ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    # ------------------------------------- OUTLIERS DETECTION ----------------------------------------
    with st.expander("🧪 Outliers Detection", expanded=False):
        st.subheader("🧪 Outliers Detection")
        outliers_iqr = outlier_dectection_iqr(picked_col) if picked_col is not None else pd.Series(dtype=float)
        outliers_z = outlier_detection_zscore_modified(picked_col, threshold=3)

        if len(outliers_iqr) == 0 and len(outliers_z) == 0:
            st.success(f"No outliers detected in {check_metric} using IQR method and Modified Z-Score method.", icon="✅")
        else:
            st.info(f"Detected {len(outliers_iqr)} outlier(s) in {check_metric} using IQR method.",
                    icon="ℹ️")
            st.dataframe(outliers_iqr.to_frame(name=f"{check_metric} Value"))
            st.info(f"Detected {len(outliers_z)} outlier(s) in {check_metric} using Modified Z-Score method.",
                    icon="🚨")
            st.dataframe(outliers_z.to_frame(name=f"{check_metric} Value"))

    #------------------------------------- HYPOTHESIS TESTING ----------------------------------------
    with st.expander("🛠️ Tests with rest days and exercise days", expanded=False):
        st.subheader("🛠️ Statistical Tests")
        if col_pvalue > 0.05:
            group1 = recovery_exercise_done[check_metric].dropna()
            group2 = recovery_exercise_notdone[check_metric].dropna()
            st.write("Since the data appears to be normally distributed, you can use parametric tests such as t-tests or ANOVA for further analysis.")
            options = ["One-sample t-test", "Independent t-test"]
            choice = st.segmented_control(
                "Select test to perform:",
                options=options, key="parametric_tests_control")
            if choice == "One-sample t-test":
                c1, c2 = st.columns(2)
                with c1:
                    option_df = st.selectbox("Select data group to test against population mean:",
                                             ["Exercise Days", "Rest Days", "All Days"])
                    if option_df == "Exercise Days":
                        group = recovery_exercise_done[check_metric].dropna()
                        st.warning("T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).", icon="⚠️")
                        popmean = st.number_input("Enter population mean to compare against:", value=float(mean_val) if mean_val is not None and not pd.isna(mean_val) else 0.0)
                        alternative = st.selectbox("Select alternative hypothesis:", ["two-sided", "less", "greater"])
                        ttest_res = stats.ttest_1samp(group, popmean, alternative=alternative)
                        button_run = st.button("Run One-sample t-test")
                        if button_run:
                            st.write(f"t-statistic: {ttest_res.statistic:.3f}," f" p-value: {ttest_res.pvalue:.3f}")
                            if ttest_res.pvalue < 0.05:
                                st.success("Reject the null hypothesis at α=0.05 level.")
                            else:
                                st.info("Fail to reject the null hypothesis at α=0.05 level.")
                    elif option_df == "Rest Days":
                        group = recovery_exercise_notdone[check_metric].dropna()
                        st.warning("T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).", icon="⚠️")
                        popmean = st.number_input("Enter population mean to compare against:", value=float(mean_val) if mean_val is not None and not pd.isna(mean_val) else 0.0)
                        alternative = st.selectbox("Select alternative hypothesis:", ["two-sided", "less", "greater"])
                        ttest_res = stats.ttest_1samp(group, popmean, alternative=alternative)
                        button_run = st.button("Run One-sample t-test")
                        if button_run:
                            st.write(f"t-statistic: {ttest_res.statistic:.3f}," f" p-value: {ttest_res.pvalue:.3f}")
                            if ttest_res.pvalue < 0.05:
                                st.success("Reject the null hypothesis at α=0.05 level.")
                            else:
                                st.info("Fail to reject the null hypothesis at α=0.05 level.")
                    elif option_df == "All Days":
                        group = recovery[check_metric].dropna()
                        st.warning("T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).", icon="⚠️")
                        popmean = st.number_input("Enter population mean to compare against:", value=float(mean_val) if mean_val is not None and not pd.isna(mean_val) else 0.0)
                        alternative = st.selectbox("Select alternative hypothesis:", ["two-sided", "less", "greater"])
                        ttest_res = stats.ttest_1samp(group, popmean, alternative=alternative)
                        button_run = st.button("Run One-sample t-test", key="one_sample_ttest_button")
                        if button_run:
                            st.write(f"t-statistic: {ttest_res.statistic:.3f}," f" p-value: {ttest_res.pvalue:.3f}")
                            if ttest_res.pvalue < 0.05:
                                st.success("Reject the null hypothesis at α=0.05 level.")
                            else:
                                st.info("Fail to reject the null hypothesis at α=0.05 level.")
                with c2:
                    fig, ax = plt.subplots(figsize=(7,5))
                    sns.kdeplot(group, color="lightblue", label=f"{option_df}", ax=ax)
                    ax.axvline(group.mean(), color="blue", linestyle="--", label=f"Exercise Mean: {group.mean():.2f}")
                    ax.axvline(popmean, color="red", linestyle="--", label=f"Population Mean: {popmean:.2f}")
                    sns.despine(ax=ax)
                    plt.title(f"Distribution of {check_metric}")
                    plt.xlabel(check_metric)
                    plt.ylabel("Density")
                    plt.legend(loc="best", fontsize=7)
                    st.pyplot(fig)

            elif choice == "Independent t-test":
                c1, c2 = st.columns(2)
                with c1:
                    st.warning("T Test are about means. They test whether the mean is different from the population mean (One-sample test) or whether the means of the groups are significantly different (Independent test).", icon="⚠️")
                    st.info(f"Group 1 is {check_metric} with exercise ({group1.shape[0]} samples)")
                    st.info(f"Group 2 is {check_metric} on rest days {group2.shape[0]} samples)")
                    st.write("Exercise mean:", float(group1.mean()))
                    st.write("Rest mean:", float(group2.mean()))
                    st.write("Δ mean (ex - rest):", float(group1.mean() - group2.mean()))
                    alternative = st.selectbox("Select alternative hypothesis:", ["two-sided", "less", "greater"])
                    ttest2_res = stats.ttest_ind(group1, group2, alternative=alternative, equal_var=False)
                    button_run2 = st.button("Run Independent t-test", key="independent_ttest_button")
                    if button_run2:
                        st.write(f"t-statistic: {ttest2_res.statistic:.3f}," f" p-value: {ttest2_res.pvalue:.3f}")
                        if ttest2_res.pvalue < 0.05:
                            st.success("Reject the null hypothesis at α=0.05 level.")
                        else:
                            st.info("Fail to reject the null hypothesis at α=0.05 level.")
                with c2:
                    fig, ax = plt.subplots(figsize=(7,5))
                    sns.kdeplot(group1, color="lightblue", label="Exercise Days", ax=ax)
                    sns.kdeplot(group2, color="salmon", label="Rest Days", ax=ax)
                    ax.axvline(group1.mean(), color="blue", linestyle="--", label=f"Exercise Mean: {group1.mean():.2f}")
                    ax.axvline(group2.mean(), color="red", linestyle="--", label=f"Rest Mean: {group2.mean():.2f}")
                    sns.despine(ax=ax)
                    plt.title(f"Distribution of {check_metric}")
                    plt.xlabel(check_metric)
                    plt.ylabel("Density")
                    plt.legend(loc="best", fontsize=7)
                    st.pyplot(fig)
        elif col_pvalue <= 0.05:
            st.write("Since the data does not appear to be normally distributed, you can use non-parametric tests such as the Wilcoxon signed-rank test or the Mann-Whitney U test for further analysis.")
            options = ["Spearman Correlation", "Mann-Whitney U test"]
            choice = st.segmented_control(
                "Select test to perform:",
                options=options)
            if choice == "Spearman Correlation":
                col2 = st.selectbox("Select another metric to correlate with:", allowed, index=allowed.index("Stress"))
                # Align pairs by dropping rows where either metric is NaN
                df_pair = recovery[[check_metric, col2]].dropna()
                st.info(f"Using {df_pair.shape[0]} paired observations for correlation.")
                npairs = df_pair.shape[0]
                if npairs < 500:
                    st.warning("Spearman correlation is accurate for large samples (over 500 samples). For smaller samples, interpret results with caution.", icon="⚠️")
                if df_pair.shape[0] < 2:
                    st.warning("Need at least 2 paired observations to compute correlation.")
                else:
                    x = df_pair[check_metric].astype(float)
                    y = df_pair[col2].astype(float)
                    try:
                        alternative = st.selectbox("Select alternative hypothesis for Spearman correlation:", ["two-sided", "less", "greater"], key="spearman_alternative_selectbox")
                        spearman_corr = stats.spearmanr(x, y, alternative=alternative)
                    except Exception as e:
                        st.error(f"Could not compute Spearman correlation: {e}")
                    else:
                        button_run = st.button("Run Spearman Correlation", key="spearman_corr_button")
                        if button_run:
                            c1, c2 = st.columns(2)
                            with c1:
                                coef_raw = spearman_corr.statistic
                                p_raw = spearman_corr.pvalue

                                coef_spearman = coef_raw[0, 1] if hasattr(coef_raw, "ndim") and coef_raw.ndim > 0 else coef_raw
                                p_value_val = p_raw[0, 1] if hasattr(p_raw, "ndim") and p_raw.ndim > 0 else p_raw

                                st.write(f"Spearman correlation coefficient: {coef_spearman}")
                                st.write(f"p-value: {p_value_val}")

                                if p_value_val < 0.05:
                                    st.success("Reject the null hypothesis at α=0.05 level.")
                                else:
                                    st.info("Fail to reject the null hypothesis at α=0.05 level.")
                            with c2:
                                fig, ax = plt.subplots(figsize=(7,5))
                                sns.scatterplot(x=x, y=y, ax=ax, alpha=0.7)
                                sns.regplot(x=x, y=y, lowess=True, scatter=False, ax=ax, color="orange")
                                ax.set_title(f"Spearman correlation (Spearman coef = {coef_spearman:.2f}, p = {p_value_val:.3f})", fontsize=14, fontweight="bold", pad=15)
                                ax.set_xlabel(check_metric)
                                ax.set_ylabel(col2)
                                sns.despine(ax=ax)
                                st.pyplot(fig)

            elif choice == "Mann-Whitney U test":
                group1 = recovery_exercise_done[check_metric].dropna()
                group2 = recovery_exercise_notdone[check_metric].dropna()
                c1, c2 = st.columns(2)
                with c1:
                    st.warning("Mann-Whitney U is about distributions, not means. It tests whether values from one group tend to be higher than the other.", icon="⚠️")
                    st.info(f"Group 1 is {check_metric} with exercise ({group1.shape[0]} samples)")
                    st.info(f"Group 2 is {check_metric} on rest days ({group2.shape[0]} samples)")
                    st.write("Exercise median:", float(group1.median()))
                    st.write("Rest median:", float(group2.median()))
                    st.write("Δ median (ex - rest):", float(group1.median() - group2.median()))
                    alternative = st.selectbox("Select alternative hypothesis:", ["two-sided", "less", "greater"])
                    button_run2 = st.button("Run Mann-Whitney U test", key="mwu_test_button")
                    if button_run2:
                        stats_mwu, pvalue_mwu = stats.mannwhitneyu(group1, group2, alternative=alternative)
                        st.write(f"U statistic: ", stats_mwu)
                        st.write(f"p-value: ", pvalue_mwu)
                        if pvalue_mwu < 0.05:
                            st.success("Reject the null hypothesis at α=0.05 level.")
                        else:
                            st.info("Fail to reject the null hypothesis at α=0.05 level.")
                        u = stats_mwu
                        n1 = len(group1)
                        n2 = len(group2)
                        # Calculate effect size
                        CLES = u / (n1 * n2)
                        st.write(f"Common Language Effect Size (CLES):", round(CLES, 2))
                        if CLES > 0.5:
                            st.write(f"{CLES*100:.1f}% chance that a randomly selected exercise day has a higher "
                                f"{check_metric} than a randomly selected rest day.")
                        elif CLES < 0.5:
                            st.write(f"{(1-CLES)*100:.1f}% chance that a randomly selected rest day has a higher "
                                f"{check_metric} than a randomly selected exercise day.")
                        else:
                            st.write("No difference between exercise and rest days in " + check_metric + ".")
                with c2:
                    fig, ax = plt.subplots(figsize=(7,5))
                    sns.kdeplot(group1, color="lightblue", label="Exercise Days", ax=ax)
                    sns.kdeplot(group2, color="salmon", label="Rest Days", ax=ax)
                    sns.despine(ax=ax)
                    plt.title(f"Distribution of {check_metric}", fontsize=14, fontweight="bold", pad=15)
                    plt.xlabel(check_metric)
                    plt.ylabel("Density")
                    plt.legend(loc="best")
                    st.pyplot(fig)
# =========================
# TAB 6 — MODELS
# =========================
with tab6:
    st.header("⚙️ Models")
    st.success(f"Overall Time Series Analysis: **{time_series_analysis}**")

    recovery = st.session_state.df_recovery.copy()
    recovery["Date"] = pd.to_datetime(recovery["Date"], errors="coerce")  # Convert to datetime
    recovery["Sleep_need_hrs"] = recovery["Sleep Need"].apply(string_to_decimal_hours)
    recovery["Efficiency"] = recovery["Efficiency"].str.replace('%', '').astype(float)
    recovery["Sleep_hr_surplus"] =  recovery["Asleep hrs"] - recovery["Sleep_need_hrs"]

    predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]
    df_model = recovery[["Date"] + predictors + ["Score", "Quality"]].dropna().copy()
    df_model = df_model.sort_values("Date")

    st.write("Modeling on: ", df_model.shape[0], "samples with no missing values in selected features and Score.")

    for col in predictors:
        if col not in df_model.columns:
            st.error(f"Predictor column '{col}' not found in data.")
            st.stop()

    types = st.selectbox(f"Select algorithm to solve:", options=["Regression", "Classification", "Unsupervised"], key=f"algorithm_type_solver")
    if types == "Regression":
        models = st.segmented_control(
            "Select Model Type:",
            ["OLS diagnosis", "Other Linear Models", "Non Linear Models", "Bagging & Boosting Models"], key="model_type_control", default="OLS diagnosis")
        if models == "OLS diagnosis":    #Linear Regression Selected
                #------------------------------FROZEN MODEL CONDITIONALS-----------------------------
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
            #------------------------------OLS LINEAR REGRESSION TRAINING PHASE-----------------------------
            if (st.session_state.model_frozen is None) and (n < 300):
                st.warning("MODEL ON TRAINING PHASE YET",icon="spinner")

                H= 40    #Test size of 40 samples
                train_lin = df_model.iloc[:-H].copy()
                test_lin  = df_model.iloc[-H:].copy()
                #train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
                X = sm.add_constant(train_lin[predictors])
                y = train_lin["Score"]
                model_linear = sm.OLS(y, X).fit(cov_type='HC3')
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
                    influential_points = st.selectbox("Include Influential Points in Model?", [True, False], key="influential_points_control")
                    c1, c2 = st.columns(2)
                    if influential_points == True:
                        with c1:
                            st.subheader("📊 Ramsey RESET Test for Linearity")
                            reset_test = linear_reset(model_linear, power=2, use_f=True)
                            st.write(f"F-statistic: {reset_test.fvalue:.3f}, p-value: {reset_test.pvalue:.3f}")
                            if reset_test.pvalue < 0.04:
                                st.warning("Reject the null hypothesis of linearity. Consider adding polynomial or interaction terms.", icon="⚠️")
                            elif reset_test.pvalue < 0.06:
                                st.info("Marginal evidence against linearity (p-value close to 0.05). Consider exploring non-linear relationships.", icon="ℹ️")
                            else:
                                st.success("Fail to reject the null hypothesis of linearity. No strong evidence of non-linearity detected.")
                            fig, axes = plt.subplots(3, 3, figsize=(14, 8))
                            for ax, col in zip(axes.flatten(), predictors):
                                ax.scatter(train_lin[col], y, alpha=0.4, s=20)
                                # Lowess reveals the true shape
                                lowess = sm.nonparametric.lowess(y, train_lin[col], frac=0.6)
                                ax.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2)
                                ax.set_title(f"Score vs {col}")
                                sns.despine(ax=ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.subheader("📊 Variance Inflation Factor (VIF)")
                            vif_data = pd.DataFrame()
                            vif_data["feature"] = X.columns
                            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                            vif_data["Meaning"] = vif_data["VIF"].apply(lambda x: "Low multicollinearity" if x < 5 else ("Moderate multicollinearity" if x < 10 else "High multicollinearity"))
                            st.dataframe(vif_data)

                            st.subheader("📊 Durbin-Watson Test for Autocorrelation")
                            dw_statistic = durbin_watson(model_linear.resid)
                            st.write(f"Durbin-Watson statistic: {dw_statistic:.3f}")
                            st.info("Durbin-Watson statistic ranges from 0 to 4. A value around 2 suggests no autocorrelation. Values < 2 indicate positive autocorrelation, while values > 2 indicate negative autocorrelation.", icon="ℹ️")
                            if dw_statistic < 1.5:
                                st.warning("Evidence of positive autocorrelation in residuals. Consider adding lagged variables or using time series models.", icon="⚠️")
                            elif dw_statistic > 2.5:
                                st.warning("Evidence of negative autocorrelation in residuals. Consider adding lagged variables or using time series models.", icon="⚠️")
                            else:
                                st.success("Durbin-Watson statistic close to 2 suggests no strong evidence of autocorrelation in residuals.")

                        with c2:
                            st.subheader("📏 Leverage")
                            influence = model_linear.get_influence()
                            leverage = influence.hat_matrix_diag
                            threshold_leverage = 2 * (X.shape[1] + 1) / X.shape[0]
                            high_leverage_points = np.where(leverage > threshold_leverage)[0]

                            st.write(f"\nHigh leverage points (leverage > {threshold_leverage:.4f}): {high_leverage_points}")
                            st.subheader("📏 Influential Observations (Cook's Distance)")
                            influence = model_linear.get_influence()
                            cooks_d, p_values = influence.cooks_distance

                            threshold = 4 / len(y)
                            influential = np.where(cooks_d > threshold)[0]
                            st.write(f"\nInfluential points (Cook's D > 4/n={threshold:.4f}): {influential}")

                            # --- Plot ---
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt='gray')
                            ax.axhline(threshold, color='red', linestyle='--', label=f"Threshold 4/n = {threshold:.4f}\nAmount of influences: {len(influential)}")

                            for i in influential:
                                ax.annotate(f"{i}", (i, cooks_d[i]), textcoords="offset points",
                                            xytext=(0, 5), fontsize=8, color='red')
                            ax.set_xlabel("Observation Index")
                            ax.set_ylabel("Cook's Distance")
                            ax.set_title("Cook's Distance — Influential Observations", fontsize=14, fontweight='bold', pad=15)
                            ax.legend()
                            sns.despine(ax=ax)
                            plt.tight_layout()
                            st.pyplot(fig)

                            df_model_influential = df_model.iloc[influential]
                            st.dataframe(df_model_influential[["Date"] + predictors + ["Score"]].reset_index(drop=True))
                    elif influential_points == False:
                        influence = model_linear.get_influence()
                        leverage = influence.hat_matrix_diag
                        threshold = 4 / len(y)
                        influence = model_linear.get_influence()
                        cooks_d, p_values = influence.cooks_distance
                        influential = np.where(cooks_d > threshold)[0]
                        df_model_no_influential = df_model.drop(df_model.index[influential])
                        train_lin = df_model_no_influential.iloc[:-H].copy()
                        test_lin  = df_model_no_influential.iloc[-H:].copy()
                        X = sm.add_constant(train_lin[predictors])
                        y = train_lin["Score"]
                        model_linear = sm.OLS(y, X).fit(cov_type='HC3')
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
                            st.write(f"F-statistic: {reset_test.fvalue:.3f}, p-value: {reset_test.pvalue:.3f}")
                            if reset_test.pvalue < 0.04:
                                st.warning("Reject the null hypothesis of linearity. Consider adding polynomial or interaction terms.", icon="⚠️")
                            elif reset_test.pvalue < 0.06:
                                st.info("Marginal evidence against linearity (p-value close to 0.05). Consider exploring non-linear relationships.", icon="ℹ️")
                            else:
                                st.success("Fail to reject the null hypothesis of linearity. No strong evidence of non-linearity detected.")
                            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
                            for ax, col in zip(axes.flatten(), predictors):
                                ax.scatter(train_lin[col], y, alpha=0.4, s=20)
                                # Lowess reveals the true shape
                                lowess = sm.nonparametric.lowess(y, train_lin[col], frac=0.6)
                                ax.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2)
                                ax.set_title(f"Score vs {col}")
                                sns.despine(ax=ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.subheader("📊 Variance Inflation Factor (VIF)")
                            vif_data = pd.DataFrame()
                            vif_data["feature"] = X.columns
                            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                            vif_data["Meaning"] = vif_data["VIF"].apply(lambda x: "Low multicollinearity" if x < 5 else ("Moderate multicollinearity" if x < 10 else "High multicollinearity"))
                            st.dataframe(vif_data)

                            st.subheader("📊 Durbin-Watson Test for Autocorrelation")
                            dw_statistic = durbin_watson(model_linear.resid)
                            st.write(f"Durbin-Watson statistic: {dw_statistic:.3f}")
                            st.info("Durbin-Watson statistic ranges from 0 to 4. A value around 2 suggests no autocorrelation. Values < 2 indicate positive autocorrelation, while values > 2 indicate negative autocorrelation.", icon="ℹ️")
                            if dw_statistic < 1.5:
                                st.warning("Evidence of positive autocorrelation in residuals. Consider adding lagged variables or using time series models.", icon="⚠️")
                            elif dw_statistic > 2.5:
                                st.warning("Evidence of negative autocorrelation in residuals. Consider adding lagged variables or using time series models.", icon="⚠️")
                            else:
                                st.success("Durbin-Watson statistic close to 2 suggests no strong evidence of autocorrelation in residuals.")

                        with c2:
                            st.warning(f"Removed {len(influential)} influential points from the model.", icon="⚠️")
                            st.subheader("📏 Leverage")
                            threshold_leverage = 2 * (X.shape[1] + 1) / X.shape[0]
                            high_leverage_points = np.where(leverage > threshold_leverage)[0]
                            st.write(f"\nHigh leverage points (leverage > {threshold_leverage:.4f}): {high_leverage_points}")

                            st.subheader("📏 Influential Observations (Cook's Distance)")
                            st.write(f"\nInfluential points (Cook's D > 4/n={threshold:.4f}): {influential}")
                            # --- Plot ---
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt='gray')
                            ax.axhline(threshold, color='red', linestyle='--', label=f"Threshold 4/n = {threshold:.4f}\nAmount of influences: {len(influential)}")

                            for i in influential:
                                ax.annotate(f"{i}", (i, cooks_d[i]), textcoords="offset points",
                                            xytext=(0, 5), fontsize=8, color='red')
                            ax.set_xlabel("Observation Index")
                            ax.set_ylabel("Cook's Distance")
                            ax.set_title("Cook's Distance — Influential Observations", fontsize=14, fontweight='bold', pad=15)
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
                        st.metric("Test R²", f"{r2_test_linear:.3f}", delta=f"{r2_test_linear - r2_train_linear:.3f}",
                                delta_color="green" if r2_test_linear > r2_train_linear else "red")
                    with c2:
                        st.metric("MSE", f"{mse_test_linear:.3f}", delta=f"{mse_test_linear - mse_train_linear:.3f}",
                                delta_color="red" if mse_test_linear > mse_train_linear else "green",
                                help="Mean Squared Error (MSE): lower values indicate better fit.\
                                    Penalizes larger errors more heavily.")
                    with c3:
                        st.metric("MAE", f"{mae_test_linear:.3f}", delta=f"{mae_test_linear - mae_train_linear:.3f}",
                                delta_color="red" if mae_test_linear > mae_train_linear else "green",
                                help="Mean Absolute Error (MAE): lower values indicate better fit.")
                    with c4:
                        st.metric("RMSE", f"{rmse_test_linear:.3f}", delta=f"{rmse_test_linear - rmse_train_linear:.3f}",
                                delta_color="red" if rmse_test_linear > rmse_train_linear else "green",
                                help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.")
                    with c5:
                        st.metric("Samples", f"{test_lin.shape[0]}", help="The last 40 samples used for testing.")
                    with c6:
                        st.metric("Test Start Date", f"{test_lin.Date.min().date()}")

                #--------------------------------------OLS INSIGHTS---------------------------------
                predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus_centered", "Sleep_hr_surplus_squared",
                            "Respiration", "Stress_sleep"]
                train_lin_pol = df_model.iloc[:-H].copy()
                test_lin_pol  = df_model.iloc[-H:].copy()
                surplus_mean = train_lin_pol["Sleep_hr_surplus"].mean()

                train_lin_pol["Sleep_hr_surplus_centered"] = train_lin_pol["Sleep_hr_surplus"] - surplus_mean
                train_lin_pol["Sleep_hr_surplus_squared"] = train_lin_pol["Sleep_hr_surplus_centered"] ** 2

                test_lin_pol["Sleep_hr_surplus_centered"] = test_lin_pol["Sleep_hr_surplus"] - surplus_mean
                test_lin_pol["Sleep_hr_surplus_squared"] = test_lin_pol["Sleep_hr_surplus_centered"] ** 2

                #train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
                X_pol = sm.add_constant(train_lin_pol[predictors])
                y_pol = train_lin_pol["Score"]
                model_linear_pol = sm.OLS(y_pol, X_pol).fit(cov_type='HC3')
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

                df_model["Predicted_Score_Linear_Pol"] = model_linear_pol.predict(sm.add_constant(df_model[predictors], has_constant="add"))
                df_model["Predicted_Score_Linear_Test_Data_Pol"] = np.nan
                df_model.loc[test_lin_pol.index, "Predicted_Score_Linear_Test_Data_Pol"] = model_linear_pol.predict(
                    sm.add_constant(test_lin_pol[predictors], has_constant="add"))
                df_model["Residuals_Linear_Pol"] = df_model["Score"] - df_model["Predicted_Score_Linear_Pol"]

                    # ----------------------------- MODEL SUMMARY & VISUALIZATIONS -----------------------------
                with st.expander("ℹ️ Model Summary", expanded=False):
                    c1, c2 = st.columns(2)
                    filtered_test = df_model.loc[test_lin_pol.index].dropna(subset=["Predicted_Score_Linear_Test_Data_Pol"])
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
                        #------------------RAMSEY RESET TEST FOR LINEARITY------------------
                        reset_test_pol = linear_reset(model_linear_pol, power=2, use_f=True)
                        st.subheader("📊 Ramsey RESET Test for Linearity (Polynomial Model)")
                        st.write(reset_test_pol)
                        st.write(f"F-statistic: {reset_test_pol.fvalue:.3f}, p-value: {reset_test_pol.pvalue:.3f}")
                        if reset_test_pol.pvalue < 0.04:
                            st.warning("Reject the null hypothesis of linearity. Consider adding higher-order polynomial or interaction terms.", icon="⚠️")
                        elif reset_test_pol.pvalue < 0.06:
                            st.info("Marginal evidence against linearity (p-value close to 0.05). Consider exploring non-linear relationships.", icon="ℹ️")
                        else:
                            st.success("Fail to reject the null hypothesis of linearity. No strong evidence of non-linearity detected.")
                    with c2:
                        fig, ax = plt.subplots(figsize=(10,5))
                        sns.lineplot(data=df_model, x="Date", y="Score", label="Actual", ax=ax, color="lightgreen", alpha=0.7)
                        sns.lineplot(data=df_model, x="Date", y="Predicted_Score_Linear_Pol", label="Predicted", ax=ax, linewidth=1, color="blue")
                        sns.lineplot(data=df_model, x="Date", y="Predicted_Score_Linear_Test_Data_Pol", label="Predicted (Test Data)", ax=ax, linestyle="--", linewidth=1, color="orange")
                        ax.axvspan(test_start, test_end, color="lightgrey", alpha=0.2, label="Test Set Period")
                        ax.set_title("Actual vs Predicted Sleep Score (Train & Test Set)", fontweight="bold", fontsize=14, pad=15)
                        ax.set_xlabel("")
                        ax.set_ylabel("Score")
                        ax.tick_params(axis='x', rotation=45)
                        ax.legend(loc="best", fontsize=7)
                        sns.despine(ax=ax)
                        st.pyplot(fig)

                        fig, ax = plt.subplots(figsize=(10,5))
                        sns.lineplot(data=filtered_test, x="Date", y="Score", label="Actual", ax=ax, marker="o", color="green", alpha=0.7)
                        sns.lineplot(data=filtered_test, x="Date", y="Predicted_Score_Linear_Test_Data_Pol", label="Predicted (Test Data)",
                                    ax=ax, marker="x", linewidth=1.5, linestyle=":", color="orange")
                        ax.set_xlabel("")
                        ax.set_ylabel("Score")
                        ax.set_title("Actual vs Predicted Sleep Score (Test Set)", fontweight="bold", fontsize=14, pad=15)
                        ax.tick_params(axis='x', rotation=45)
                        sns.despine(ax=ax)
                        ax.legend(loc="lower left", fontsize=7)
                        st.pyplot(fig)
                        st.markdown(f"**Out-of-sample Test R² (trained on train set):** {r2_test_linear_pol:.3f}")

                        fig, ax = plt.subplots(figsize=(10,5))
                        sns.scatterplot(data=filtered_test, x="Score", y="Predicted_Score_Linear_Test_Data_Pol", ax=ax, color="purple", alpha=0.7,
                                        hue="Quality", palette="viridis", legend="full")
                        sns.lineplot(data=filtered_test, x="Score", y="Score", ax=ax, color="red", linestyle="--", label="Ideal Fit")
                        ax.axvline(x=80, color="grey", linestyle=":", label="Good Quality Threshold")
                        ax.axvline(x=90, color="grey", linestyle=":", label="Excellent Quality Threshold")
                        ax.set_title("Predicted vs Actual Sleep Score (Test Set)", fontweight="bold", fontsize=14, pad=15)
                        ax.set_xlabel("Actual Sleep Score")
                        ax.set_ylabel("Predicted Sleep Score")
                        sns.despine(ax=ax)
                        ax.legend(loc="lower right", fontsize=7)
                        st.pyplot(fig)
                        #-----------------------------VIF ANALYSIS-----------------------------
                        st.subheader("📊 Variance Inflation Factor (VIF) for Polynomial Model")
                        vif_data_pol = pd.DataFrame()
                        vif_data_pol["feature"] = X_pol.columns
                        vif_data_pol["VIF"] = [variance_inflation_factor(X_pol.values, i) for i in range(X_pol.shape[1])]
                        vif_data_pol["Meaning"] = vif_data_pol["VIF"].apply(lambda x: "Low multicollinearity" if x < 5 else ("Moderate multicollinearity" if x < 10 else "High multicollinearity"))
                        st.dataframe(vif_data_pol)
                        #-----------------------------DURBIN-WATSON TEST-----------------------------
                        st.subheader("📊 Durbin-Watson Test for Autocorrelation (Polynomial Model)")
                        dw_statistic_pol = durbin_watson(model_linear_pol.resid)
                        st.write(f"Durbin-Watson statistic: {dw_statistic_pol:.3f}")
                        st.info("Durbin-Watson statistic ranges from 0 to 4. A value around 2 suggests no autocorrelation. Values < 2 indicate positive autocorrelation, while values > 2 indicate negative autocorrelation.", icon="ℹ️")
                        if dw_statistic_pol < 1.5:
                            st.warning("Evidence of positive autocorrelation in residuals. Consider adding lagged variables or using time series models.", icon="⚠️")
                        elif dw_statistic_pol > 2.5:
                            st.warning("Evidence of negative autocorrelation in residuals. Consider adding lagged variables or using time series models.", icon="⚠️")
                        else:
                            st.success("Durbin-Watson statistic close to 2 suggests no strong evidence of autocorrelation in residuals.")
                    #----------------------------- PERFORMANCE METRICS -----------------------------
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
                        st.metric("Test R²", f"{r2_test_linear_pol:.3f}", delta=f"{r2_test_linear_pol - r2_train_linear_pol:.3f}",
                                delta_color="green" if r2_test_linear_pol > r2_train_linear_pol else "red")
                    with c2:
                        st.metric("Test MSE", f"{mse_test_linear_pol:.3f}", delta=f"{mse_test_linear_pol - mse_train_linear_pol:.3f}",
                                delta_color="red" if mse_test_linear_pol > mse_train_linear_pol else "green",
                                help="Mean Squared Error (MSE): lower values indicate better fit.\
                                    Penalizes larger errors more heavily.")
                    with c3:
                        st.metric("Test MAE", f"{mae_test_linear_pol:.3f}", delta=f"{mae_test_linear_pol - mae_train_linear_pol:.3f}",
                                delta_color="red" if mae_test_linear_pol > mae_train_linear_pol else "green",
                                help="Mean Absolute Error (MAE): lower values indicate better fit.")
                    with c4:
                        st.metric("Test RMSE", f"{rmse_test_linear_pol:.3f}", delta=f"{rmse_test_linear_pol - rmse_train_linear_pol:.3f}",
                                delta_color="red" if rmse_test_linear_pol > rmse_train_linear_pol else "green",
                                help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.")
                    with c5:
                        st.metric("Test Samples", f"{test_lin_pol.shape[0]}", help="The last 40 samples used for testing.")
                    with c6:
                        st.metric("Test Start Date", f"{test_lin_pol.Date.min().date()}")
                    # ----------------------------- PREDICTIONS DATAFRAME -----------------------------
                    with st.expander("🗂️ Predictions Dataframe", expanded=False):
                            st.dataframe(df_model[["Date", "Score", "Predicted_Score_Linear_Pol", "Predicted_Score_Linear_Test_Data_Pol", "Residuals_Linear_Pol"]].sort_values("Date"))

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
                        learning_curve_df["% Δ RMSE"] = (learning_curve_df["Test RMSE"].diff().fillna(0)*100) / learning_curve_df["Test RMSE"].shift(1).replace(0, np.nan)
                        st.dataframe(learning_curve_df[["Test RMSE","% Δ RMSE"]])
                        st.write("k=", 3)
                        st.write("% Δ no greater than", 5)

                        #----------------------------- PLOTTING LEARNING CURVE -----------------------------
                    with c2:
                        st.subheader("📈 Learning Curve Plot")
                        metric = st.selectbox("Select metric to plot:", ["MAE", "MSE", "RMSE"], index=2, key="learning_curve_metric_selectbox")
                        samples = st.checkbox("Show all values for learning curve (not forecast or extrapolated) ?:", value=True, key="learning_curve_future_values_checkbox")
                        fig, ax = plt.subplots(figsize=(10,5))
                        if samples:
                            sns.lineplot(data=learning_curve_df, x=learning_curve_df.index, y=f"Train {metric}",  label=f"Train {metric}", ax=ax, color="lightblue", linestyle=":", linewidth=1, marker="o", markersize=4)
                            sns.lineplot(data=learning_curve_df, x=learning_curve_df.index, y=f"Test {metric}",  label=f"Test {metric}", ax=ax, color="orange", linewidth=1, marker="x", markersize=4)
                            ax.axvspan(xmin=40, xmax=n, color="lightgrey", alpha=0.2, label="Current Region")
                        else:
                            filtered_lc = learning_curve_df.loc[learning_curve_df.index <= n]
                            sns.lineplot(data=filtered_lc, x=filtered_lc.index, y=f"Train {metric}", label=f"Train {metric}", ax=ax, color="lightblue", linestyle=":", linewidth=1, marker="o", markersize=2)
                            sns.lineplot(data=filtered_lc, x=filtered_lc.index, y=f"Test {metric}", label=f"Test {metric}", ax=ax, color="orange", linewidth=1, marker="x")

                        ax.axvline(x=n, color="white", linestyle="--", label="Current Sample Size")
                        ax.set_title(f"Learning Curve: {metric} vs Training Size", fontweight="bold", fontsize=14, pad=15)
                        ax.set_xlabel("Model Samples")
                        ax.set_ylabel(metric)
                        ax.tick_params(axis='x', rotation=45)
                        ax.legend(loc="best", fontsize=7)
                        sns.despine(ax=ax)
                        st.pyplot(fig)

                        for params in model_linear_pol.params.index:
                            if model_linear_pol.pvalues[params] < 0.05:
                                st.success(f"{params} Coeff: {model_linear_pol.params[params]:.4f} P-value: {model_linear_pol.pvalues[params]:.4f} (Significant at α=0.05)" )
                            else:
                                st.warning(f"{params} Coeff: {model_linear_pol.params[params]:.4f} P-value: {model_linear_pol.pvalues[params]:.4f}" )

                #----------------------------- RESIDUALS ANALYSIS -----------------------------
                with st.expander("📊 Residuals Analysis", expanded=False):
                    option_residuals = ["On Train Set", "On Test Set", "Both"]
                    residuals_analysis = st.segmented_control("Select residuals analysis dataset:", options=option_residuals, key="residuals_analysis")
                    if residuals_analysis == "On Train Set":
                        st.subheader("🧾 Residuals Analysis on Train Set")
                        residuals_train_df = df_model[df_model.index.isin(train_lin_pol.index)].copy()
                        residuals_train_df["Residuals"] = model_linear_pol.resid
                        fitted = model_linear_pol.fittedvalues
                        pvalue_resid, interpretation_resid = normality_test(residuals_train_df["Residuals"])
                        if pvalue_resid > 0.05:
                            st.success(f"Residuals appear to be normally distributed (p={pvalue_resid:.3f}).")
                        else:
                            st.info(f"Residuals do not appear to be normally distributed (p={pvalue_resid:.3f}).")
                        st.dataframe(fit_distribution(residuals_train_df["Residuals"]))
                        st.subheader(" Breusch-Pagan Test for Heteroscedasticity")
                        bp_test = het_breuschpagan(model_linear_pol.resid, model_linear_pol.model.exog)
                        bp_labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
                        bp_results = dict(zip(bp_labels, bp_test))
                        if bp_results['p-value'] < 0.05:
                            st.warning(f"❌ Evidence of heteroscedasticity (p={bp_results['p-value']:.3f}). Consider using robust standard errors or transforming variables.", icon="⚠️")
                        else:
                            st.success(f"✅ No evidence of heteroscedasticity (p={bp_results['p-value']:.3f}).")
                        c1, c2 = st.columns(2)
                        with c1:
                            fig, ax = plt.subplots(figsize=(8,4))
                            sm.qqplot(residuals_train_df["Residuals"], line='45', ax=ax, fit=True)
                            ax.set_title("QQ Plot of Residuals (Train Set)", fontsize=14, fontweight="bold", pad=15)
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        with c2:
                            fig, ax = plt.subplots(figsize=(8,4))
                            sns.histplot(residuals_train_df["Residuals"], kde=True, stat="density", ax=ax)
                            ax.set_title("Histogram of Residuals (Train Set)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("Residuals")
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        c3, c4 = st.columns(2)
                        with c3:
                            fig, ax = plt.subplots(figsize=(8,4))
                            sns.scatterplot(x=fitted, y=residuals_train_df["Residuals"], ax=ax)
                            ax.axhline(0, color="red", linestyle="--")
                            ax.set_title("Residuals vs Fitted Values (Train Set)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("Fitted Values")
                            ax.set_ylabel("Residuals")
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        with c4:
                            fig, ax = plt.subplots(figsize=(8,4))
                            sns.lineplot(data=residuals_train_df, x="Date", y="Residuals", ax=ax)
                            ax.axhline(0, color="red", linestyle="--")
                            ax.set_title("Residuals over Time (Train Set)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("")
                            ax.set_ylabel("Residuals")
                            ax.tick_params(axis='x', rotation=45)
                            sns.despine(ax=ax)
                            st.pyplot(fig)

                    elif residuals_analysis == "On Test Set":
                        st.subheader("🧾 Residuals Analysis on Test Set")
                        residuals_test_df = df_model[df_model.index.isin(test_lin_pol.index)].copy()
                        residuals_test_df["Residuals"] = residuals_test_df["Score"] - residuals_test_df["Predicted_Score_Linear_Test_Data_Pol"]
                        fitted_test = residuals_test_df["Predicted_Score_Linear_Test_Data_Pol"]
                        pvalue_resid_test, interpretation_resid_test = normality_test(residuals_test_df["Residuals"])
                        if pvalue_resid_test > 0.05:
                            st.success(f"Residuals appear to be normally distributed (p={pvalue_resid_test:.3f}).")
                        else:
                            st.info(f"Residuals do not appear to be normally distributed (p={pvalue_resid_test:.3f}).")

                        c1, c2= st.columns(2)
                        with c1:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sm.qqplot(residuals_test_df["Residuals"], line='45', ax=ax, fit=True)
                            ax.set_title("QQ Plot of Residuals (Test Set)", fontsize=14, fontweight="bold", pad=15)
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        with c2:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sns.histplot(residuals_test_df["Residuals"], kde=True, stat="density", ax=ax)
                            ax.set_title("Histogram of Residuals (Test Set)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("Residuals")
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        c3, c4 = st.columns(2)
                        with c3:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sns.scatterplot(x=fitted_test, y=residuals_test_df["Residuals"], ax=ax)
                            ax.axhline(0, color="red", linestyle="--")
                            ax.set_title("Residuals vs Fitted Values (Test Set)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("Fitted Values")
                            ax.set_ylabel("Residuals")
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        with c4:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sns.lineplot(data=residuals_test_df, x="Date", y="Residuals", ax=ax)
                            ax.axhline(0, color="red", linestyle="--")
                            ax.set_title("Residuals over Time (Test Set)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("")
                            ax.set_ylabel("Residuals")
                            ax.tick_params(axis='x', rotation=45)
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                    elif residuals_analysis == "Both":
                        st.subheader("🧾 Residuals Analysis on Train & Test Set")
                        df_model["Residuals"] = df_model["Score"] - df_model["Predicted_Score_Linear_Pol"]
                        pvalue_resid_both, interpretation_resid_both = normality_test(df_model["Residuals"])
                        if pvalue_resid_both > 0.05:
                            st.success(f"Residuals appear to be normally distributed (p={pvalue_resid_both:.3f}).")
                        else:
                            st.info(f"Residuals do not appear to be normally distributed (p={pvalue_resid_both:.3f}).")

                        c1, c2 = st.columns(2)
                        with c1:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sm.qqplot(df_model["Residuals"], line='45', ax=ax, fit=True)
                            ax.set_title("QQ Plot of Residuals (All Data)", fontsize=14, fontweight="bold", pad=15)
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        with c2:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sns.histplot(df_model["Residuals"], kde=True, stat="density", ax=ax)
                            ax.set_title("Histogram of Residuals (All Data)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("Residuals")
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        c3, c4 = st.columns(2)
                        with c3:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sns.scatterplot(x=df_model["Predicted_Score_Linear_Pol"], y=df_model["Residuals"], ax=ax)
                            ax.axhline(0, color="red", linestyle="--")
                            ax.set_title("Residuals vs Fitted Values (All Data)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("Fitted Values")
                            ax.set_ylabel("Residuals")
                            sns.despine(ax=ax)
                            st.pyplot(fig)
                        with c4:
                            fig, ax = plt.subplots(figsize=(10,5))
                            sns.lineplot(data=df_model, x="Date", y="Residuals", ax=ax)
                            ax.axhline(0, color="red", linestyle="--")
                            ax.set_title("Residuals over Time (All Data)", fontsize=14, fontweight="bold", pad=15)
                            ax.set_xlabel("")
                            ax.set_ylabel("Residuals")
                            ax.tick_params(axis='x', rotation=45)
                            sns.despine(ax=ax)
                            st.pyplot(fig)

            # ------------------------------FROZEN MODEL DEPLOYMENT PHASE-----------------------------
            elif (st.session_state.model_frozen is None) and (n >= 300):
                st.success("MODEL READY FOR DEPLOYMENT (FREEZING NOW)", icon="✅")
                freeze_date = df_model.iloc[199]["Date"]  # Freeze after first 200 samples (0-199)
                frozen_df = df_model[df_model["Date"] <= freeze_date].copy()

                X_frozen = sm.add_constant(frozen_df[predictors])
                y_frozen = frozen_df["Score"]

                model_frozen = sm.OLS(y_frozen, X_frozen).fit(cov_type='HC3')

                st.session_state.model_frozen = model_frozen
                st.session_state.freeze_date = freeze_date
                st.session_state.freeze_predictors = predictors.copy()

                st.info(f"Frozen on {freeze_date.date()} New data after this will be monitored, not used for training.", icon="ℹ️")
            #------------------------------ DEPLOYMENT & MONITORING -----------------------------
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
                        fig, ax = plt.subplots(figsize=(10,4))
                        sns.lineplot(data=live_df, x="Date", y="resid_frozen", ax=ax)
                        ax.axhline(0, color="red", linestyle="--")
                        ax.set_title("Post-freeze residuals over time")
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)


        elif models == "Other Linear Models":
            @st.cache_data(show_spinner="Fitting linear models... (runs once per dataset)")
            def fit_reg_linear_models(X_train, y_train, X_test, y_test):

                """
                Fit OLS, Ridge, Lasso, and ElasticNet regression models with polynomial features and scaling.
                ------------
                Parameters:
                X_train: pd.DataFrame - Training features
                y_train: pd.Series - Training target
                X_test: pd.DataFrame - Testing features
                y_test: pd.Series - Testing target
                ------------
                Returns:
                dict - Dictionary of model name to fitted model and performance metrics
                """
                results = []
                # ------------------OLS with Polynomial Features------------------
                pipe_ols = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2)),
                    ("ols", LinearRegression())
                ])
                best_ols = pipe_ols.fit(X_train, y_train)
                y_pred_train_ols = best_ols.predict(X_train)
                y_pred_test_ols = best_ols.predict(X_test)
                #------------------ Ridge
                pipe_ridge = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2)),
                    ("ridge", Ridge(alpha=1.0))
                ])
                grid_ridge = GridSearchCV(pipe_ridge, param_grid={"ridge__alpha": np.logspace(-3, 3, 100)}, cv=5, scoring="neg_mean_squared_error")
                grid_ridge.fit(X_train, y_train)
                best_ridge = grid_ridge.best_estimator_
                y_pred_train_ridge = best_ridge.predict(X_train)
                y_pred_test_ridge = best_ridge.predict(X_test)

                # ------------------ Lasso
                pipe_lasso = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2)),
                    ("lasso", Lasso(alpha=0.1, max_iter=10000))
                ])
                grid_lasso = GridSearchCV(pipe_lasso, param_grid={"lasso__alpha": np.logspace(-3, 3, 100)}, cv=5, scoring="neg_mean_squared_error")
                grid_lasso.fit(X_train, y_train)
                best_lasso = grid_lasso.best_estimator_
                y_pred_train_lasso = best_lasso.predict(X_train)
                y_pred_test_lasso = best_lasso.predict(X_test)

                # ------------------ ElasticNet
                pipe_enet = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2)),
                    ("enet", ElasticNet(max_iter=10000))
                ])
                grid_enet = GridSearchCV(pipe_enet, param_grid={"enet__alpha": np.logspace(-3, 3, 100), "enet__l1_ratio": np.linspace(0.1, 1, 50)}, cv=5, scoring="neg_mean_squared_error")
                grid_enet.fit(X_train, y_train)
                best_enet = grid_enet.best_estimator_
                y_pred_train_enet = best_enet.predict(X_train)
                y_pred_test_enet = best_enet.predict(X_test)

                # ------------------- Compile results

                results.append({
                    "Model": "OLS with Polynomial Features",
                    "Train R²": r2_score(y_train, y_pred_train_ols),
                    "Test R²": r2_score(y_test, y_pred_test_ols),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_ols),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_ols),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_ols)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_ols)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_ols),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_ols)
                })
                results.append({
                    "Model": f"Ridge (alpha={grid_ridge.best_params_['ridge__alpha']:.4f})",
                    "Train R²": r2_score(y_train, y_pred_train_ridge),
                    "Test R²": r2_score(y_test, y_pred_test_ridge),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_ridge),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_ridge),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_ridge)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_ridge),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_ridge)
                })
                results.append({
                    "Model": f"Lasso (alpha={grid_lasso.best_params_['lasso__alpha']:.4f})",
                    "Train R²": r2_score(y_train, y_pred_train_lasso),
                    "Test R²": r2_score(y_test, y_pred_test_lasso),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_lasso),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_lasso),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_lasso)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_lasso)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_lasso),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_lasso)
                })
                results.append({
                    "Model": f"ElasticNet (alpha={grid_enet.best_params_['enet__alpha']:.4f}, l1_ratio={grid_enet.best_params_['enet__l1_ratio']:.2f})",
                    "Train R²": r2_score(y_train, y_pred_train_enet),
                    "Test R²": r2_score(y_test, y_pred_test_enet),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_enet),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_enet),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_enet)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_enet)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_enet),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_enet)
                })

                model_objects = [best_ols, best_ridge, best_lasso, best_enet]
                results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True)
                best_model_obj = model_objects[results_df.index[0]]

                return best_ols, best_ridge, best_lasso, best_enet, results_df, best_model_obj

            H= 40    #Test size of 40 samples
            train_lin = df_model.iloc[:-H].copy()
            test_lin  = df_model.iloc[-H:].copy()
            #train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]

            time_start = time.time()
            ols,ridge, lasso, enet, results, best_linear = fit_reg_linear_models(train_lin[predictors], train_lin["Score"], test_lin[predictors], test_lin["Score"])
            time_end = time.time()
            st.header(f"📊 Results Dataframe")
            st.dataframe(results)
            st.badge(f"Best model: {results.iloc[0]['Model']}", color="green", icon="✅")
            st.info(f"Time taken to fit models: {time_end - time_start:.2f} seconds")

            # ----------------------------- BEST MODEL PERFORMANCE -----------------------------
            st.header(f"📊 Performance of Linear Models")
            y_pred_train_best = best_linear.predict(train_lin[predictors])
            y_pred_test_best = best_linear.predict(test_lin[predictors])
            r2_train_linear_pol = r2_score(train_lin["Score"], y_pred_train_best)
            r2_test_linear_pol = r2_score(test_lin["Score"], y_pred_test_best)
            rmse_train_linear_pol = np.sqrt(mean_squared_error(train_lin["Score"], y_pred_train_best))
            rmse_test_linear_pol = np.sqrt(mean_squared_error(test_lin["Score"], y_pred_test_best))
            mae_train_linear_pol = mean_absolute_error(train_lin["Score"], y_pred_train_best)
            mae_test_linear_pol = mean_absolute_error(test_lin["Score"], y_pred_test_best)
            mse_train_linear_pol = rmse_train_linear_pol ** 2
            mse_test_linear_pol = rmse_test_linear_pol ** 2

            #----------------------------- PERFORMANCE METRICS -----------------------------
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
                    st.metric("Test R²", f"{r2_test_linear_pol:.3f}", delta=f"{r2_test_linear_pol - r2_train_linear_pol:.3f}",
                            delta_color="green" if r2_test_linear_pol > r2_train_linear_pol else "red")
                with c2:
                    st.metric("Test MSE", f"{mse_test_linear_pol:.3f}", delta=f"{mse_test_linear_pol - mse_train_linear_pol:.3f}",
                                    delta_color="red" if mse_test_linear_pol > mse_train_linear_pol else "green",
                                    help="Mean Squared Error (MSE): lower values indicate better fit.\
                                        Penalizes larger errors more heavily.")
                with c3:
                    st.metric("Test MAE", f"{mae_test_linear_pol:.3f}", delta=f"{mae_test_linear_pol - mae_train_linear_pol:.3f}",
                                    delta_color="red" if mae_test_linear_pol > mae_train_linear_pol else "green",
                                    help="Mean Absolute Error (MAE): lower values indicate better fit.")
                with c4:
                    st.metric("Test RMSE", f"{rmse_test_linear_pol:.3f}", delta=f"{rmse_test_linear_pol - rmse_train_linear_pol:.3f}",
                                    delta_color="red" if rmse_test_linear_pol > rmse_train_linear_pol else "green",
                                    help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.")
                with c5:
                    st.metric("Test Samples", f"{test_lin.shape[0]}", help="The last 40 samples used for testing.")
                with c6:
                    st.metric("Test Start Date", f"{test_lin.Date.min().date()}")
            # ----------------------------- LEARNING CURVE -----------------------------
            with st.expander("📈 Learning Curve Analysis for Best Linear Model", expanded=True):
                st.header("📈 Learning Curve for Best Linear Model")

                def metrics_lcv(df, x_train, x_test, y_train, y_test, model=best_linear):
                    """
                    Learning Curve for winner model
                    ------------
                    Parameters:
                    df: pd.DataFrame - Full dataframe
                    x_train: pd.DataFrame - Training features
                    x_test: pd.DataFrame - Testing features
                    y_train: pd.Series - Training target
                    y_test: pd.Series - Testing target
                    models: sklearn estimator - Fitted model to evaluate learning curve
                    ------------
                    Returns:
                    None - Displays learning curve plot
                    """
                    from sklearn.base import clone

                    train_sizes = np.linspace(10, len(x_train), 10).astype(int)
                    train_rmse_list = []
                    test_rmse_list  = []

                    for ts in train_sizes:
                        m = clone(model)
                        m.fit(x_train.iloc[:ts], y_train.iloc[:ts])
                        train_rmse_list.append(np.sqrt(mean_squared_error(y_train.iloc[:ts], m.predict(x_train.iloc[:ts]))))
                        test_rmse_list.append(np.sqrt(mean_squared_error(y_test, m.predict(x_test))))

                    train_mse_mean = np.array(train_rmse_list)
                    test_mse_mean  = np.array(test_rmse_list)
                    lc_df = pd.DataFrame({
                        "Train Size": train_sizes,
                        "Train Score": train_mse_mean,
                        "Test Score": test_mse_mean
                    })
                    lc_df["Gap"] = lc_df["Test Score"] - lc_df["Train Score"]

                    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
                    last_step = list(model.named_steps.values())[-1]
                    ax[0].plot(train_sizes, train_mse_mean, label="Train RMSE", color="steelblue")
                    ax[0].plot(train_sizes, test_mse_mean, label="Test RMSE (holdout)", color="orange")
                    ax[0].set_title(f"Learning Curve — {last_step.__class__.__name__}", fontweight="bold", fontsize=8, pad=15)
                    ax[0].set_xlabel("Training Set Size")
                    ax[0].set_ylabel("RMSE")
                    sns.despine(ax=ax[0])
                    ax[0].legend()
                    ax[0].grid()

                    sns.barplot(x=["Train", "Test"], y=[train_mse_mean.mean(), test_mse_mean.mean()], ax=ax[1], palette=["lightblue", "orange"])
                    ax[1].set_title(f"Average RMSE at Different Training Sizes", fontweight="bold", fontsize=8, pad=15)
                    ax[1].set_ylabel("RMSE")
                    sns.despine(ax=ax[1])
                    ax[1].grid(axis="y")

                    sns.lineplot(x=lc_df["Train Size"], y=lc_df["Gap"], marker="o", color="coral", label="RMSE gap Train - Test", ax=ax[2])
                    ax[2].set_title(f"Gap Between Train and Test RMSE ", fontweight="bold", fontsize=8, pad=15)
                    ax[2].set_xlabel("Training Set Size")
                    ax[2].set_ylabel(f"RMSE Score Gap (Train - Test)")
                    ax[2].annotate(
                        text=f"Min Gap =\n{lc_df['Gap'].min():.4f}",
                        xy=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()], lc_df["Gap"].min()),
                        xytext=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()] + 10, lc_df["Gap"].min() + 0.05),
                        textcoords="data",
                        arrowprops= dict(arrowstyle="->", color="black"),
                        fontsize=6, fontweight="bold"
                    )
                    ax[2].set_ylim(lc_df["Gap"].min() - 0.02, lc_df["Gap"].max() + 0.05)
                    sns.despine(ax=ax[2])
                    ax[2].legend(loc="upper right")

                    st.pyplot(fig)

                metrics_lcv(df_model, train_lin[predictors], test_lin[predictors], train_lin["Score"], test_lin["Score"], model=best_linear)

            # --------------------------- EXPLANATORY POWER -----------------------------
            with st.expander("📊 Explanatory Power of Predictors", expanded=False):
                st.subheader("📊 Explanatory Power of Predictors")
                 # Background dataset — small sample is enough for speed
                X_background = shap.sample(train_lin[predictors], 50)

                explainer   = shap.Explainer(best_linear.predict, X_background)
                shap_values = explainer(test_lin[predictors])
                col1, col2 = st.columns(2)
                with col1:
                    # Beeswarm
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(fig)
                with col2:
                    # Waterfall
                    sample_ind = 0
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values[sample_ind], max_display=14, show=False)
                    st.pyplot(fig)

        # ---------------------------------- NON LINEAR MODELS ----------------------------------
        elif models == "Non Linear Models":
            @st.cache_data(show_spinner="Fitting non-linear models... (runs once per dataset)")
            def fit_reg_non_linear_models(X_train, y_train, X_test, y_test):
                """
                Fit DT, SVM, KNN with scaling features and select best hyperparameters using GridSearchCV.
                ------------
                Parameters:
                X_train: pd.DataFrame - Training features
                y_train: pd.Series - Training target
                X_test: pd.DataFrame - Testing features
                y_test: pd.Series - Testing target
                ------------
                Returns:
                dict - Dictionary of model name to fitted model and performance metrics
                """
                results = []
                # ------------------DT------------------
                full_tree = DecisionTreeRegressor()
                path = full_tree.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas = path.ccp_alphas
                samples_ccp = np.quantile(ccp_alphas, np.linspace(0, 1, 10))
                pipe_dt = Pipeline([
                    ("dt", DecisionTreeRegressor(criterion="squared_error"))
                ])
                grid_dt = GridSearchCV(pipe_dt,
                                       param_grid={
                                           "dt__max_depth": [3, 5, 7, 9, None],
                                           "dt__min_samples_split": [2, 5, 10],
                                           "dt__ccp_alpha": samples_ccp
                                        },
                                        cv=5, scoring="neg_mean_squared_error")
                grid_dt.fit(X_train, y_train)
                best_dt = grid_dt.best_estimator_
                y_pred_train_dt = best_dt.predict(X_train)
                y_pred_test_dt = best_dt.predict(X_test)

                #------------------ KNN ---------------------------
                pipe_knn = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsRegressor())
                ])
                grid_knn = GridSearchCV(pipe_knn,
                                        param_grid={
                                            "knn__n_neighbors": [1, 3, 5, 7, 10, 15, 20, 30],
                                            "knn__weights":["uniform", "distance"],
                                            "knn__p": [1, 2]},
                                        cv=5, scoring="neg_mean_squared_error")
                grid_knn.fit(X_train, y_train)
                best_knn = grid_knn.best_estimator_
                y_pred_train_knn = best_knn.predict(X_train)
                y_pred_test_knn = best_knn  .predict(X_test)

                # ------------------ SVM ---------------------------
                from sklearn.svm import SVR
                pipe_svmr = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svmr", SVR())
                ])
                grid_svmr = GridSearchCV(pipe_svmr,
                                        param_grid = [
                        {   # linear — no gamma, no degree
                            "svmr__kernel": ["linear"],
                            "svmr__C":      np.logspace(-3, 3, 30),
                        },
                        {   # rbf and sigmoid — gamma applies, degree does not
                            "svmr__kernel": ["rbf", "sigmoid"],
                            "svmr__C":      np.logspace(-3, 3, 30),
                            "svmr__gamma":  ["scale", "auto"],
                        },
                        {   # poly — both gamma and degree apply
                            "svmr__kernel": ["poly"],
                            "svmr__C":      np.logspace(-3, 3, 10),
                            "svmr__gamma":  ["scale", "auto"],
                            "svmr__degree": [2, 3],
                        },
                    ], cv=5, scoring="neg_mean_squared_error")
                grid_svmr.fit(X_train, y_train)
                best_svmr = grid_svmr.best_estimator_
                y_pred_train_svmr = best_svmr.predict(X_train)
                y_pred_test_svmr = best_svmr.predict(X_test)

                # ------------------- Compile results
                results.append({
                    "Model": f"Decision Tree (ccp_alpha={grid_dt.best_params_['dt__ccp_alpha']}), max_depth={grid_dt.best_params_['dt__max_depth']}, min_samples_split={grid_dt.best_params_['dt__min_samples_split']}",
                    "Train R²": r2_score(y_train, y_pred_train_dt),
                    "Test R²": r2_score(y_test, y_pred_test_dt),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_dt),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_dt),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_dt)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_dt)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_dt),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_dt)
                })
                results.append({
                    "Model": f"KNN (n_neighbors={grid_knn.best_params_['knn__n_neighbors']}, weights={grid_knn.best_params_['knn__weights']}, p={grid_knn.best_params_['knn__p']})",
                    "Train R²": r2_score(y_train, y_pred_train_knn),
                    "Test R²": r2_score(y_test, y_pred_test_knn),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_knn),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_knn),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_knn)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_knn)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_knn),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_knn)
                })
                results.append({
                    "Model": f"SVM Regressor (kernel={grid_svmr.best_params_['svmr__kernel']}",
                    "Train R²": r2_score(y_train, y_pred_train_svmr),
                    "Test R²": r2_score(y_test, y_pred_test_svmr),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_svmr),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_svmr),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_svmr)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_svmr)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_svmr),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_svmr)
                })

                model_objects = [best_dt, best_knn, best_svmr]
                results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True)
                best_model_obj = model_objects[results_df.index[0]]

                return best_dt, best_knn, best_svmr, results_df, best_model_obj

            H= 40    #Test size of 40 samples
            train_lin = df_model.iloc[:-H].copy()
            test_lin  = df_model.iloc[-H:].copy()
            #train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]

            time_start = time.time()
            dt, knn, svmr, results, best_model_non_linear = fit_reg_non_linear_models(train_lin[predictors], train_lin["Score"], test_lin[predictors], test_lin["Score"])
            time_end = time.time()
            st.header(f"📊 Results Dataframe")
            st.dataframe(results)
            st.badge(f"Best model: {results.iloc[0]['Model']}", color="green", icon="✅")
            st.badge(f"Best parameters: {best_model_non_linear.get_params()}", icon="✅")
            st.info(f"Time taken to fit models: {time_end - time_start:.2f} seconds")

            # ----------------------------- BEST MODEL PERFORMANCE -----------------------------
            st.header(f"📊 Performance of Non-Linear Models")
            y_pred_train_best_nonlinear = best_model_non_linear.predict(train_lin[predictors])
            y_pred_test_best_nonlinear = best_model_non_linear.predict(test_lin[predictors])
            r2_train_nonlinear = r2_score(train_lin["Score"], y_pred_train_best_nonlinear)
            r2_test_nonlinear = r2_score(test_lin["Score"], y_pred_test_best_nonlinear)
            rmse_train_nonlinear = np.sqrt(mean_squared_error(train_lin["Score"], y_pred_train_best_nonlinear))
            rmse_test_nonlinear = np.sqrt(mean_squared_error(test_lin["Score"], y_pred_test_best_nonlinear))
            mae_train_nonlinear = mean_absolute_error(train_lin["Score"], y_pred_train_best_nonlinear)
            mae_test_nonlinear = mean_absolute_error(test_lin["Score"], y_pred_test_best_nonlinear)
            mse_train_nonlinear = rmse_train_nonlinear ** 2
            mse_test_nonlinear = rmse_test_nonlinear ** 2
    #----------------------------- PERFORMANCE METRICS -----------------------------
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
                    st.metric("Test R²", f"{r2_test_nonlinear:.3f}", delta=f"{r2_test_nonlinear - r2_train_nonlinear:.3f}",
                            delta_color="green" if r2_test_nonlinear > r2_train_nonlinear else "red")
                with c2:
                    st.metric("Test MSE", f"{mse_test_nonlinear:.3f}", delta=f"{mse_test_nonlinear - mse_train_nonlinear:.3f}",
                                    delta_color="red" if mse_test_nonlinear > mse_train_nonlinear else "green",
                                    help="Mean Squared Error (MSE): lower values indicate better fit.\
                                        Penalizes larger errors more heavily.")
                with c3:
                    st.metric("Test MAE", f"{mae_test_nonlinear:.3f}", delta=f"{mae_test_nonlinear - mae_train_nonlinear:.3f}",
                                    delta_color="red" if mae_test_nonlinear > mae_train_nonlinear else "green",
                                    help="Mean Absolute Error (MAE): lower values indicate better fit.")
                with c4:
                    st.metric("Test RMSE", f"{rmse_test_nonlinear:.3f}", delta=f"{rmse_test_nonlinear - rmse_train_nonlinear:.3f}",
                                    delta_color="red" if rmse_test_nonlinear > rmse_train_nonlinear else "green",
                                    help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.")
                with c5:
                    st.metric("Test Samples", f"{test_lin.shape[0]}", help="The last 40 samples used for testing.")
                with c6:
                    st.metric("Test Start Date", f"{test_lin.Date.min().date()}")

            # ----------------------------- LEARNING CURVE -----------------------------
            with st.expander("📈 Learning Curve Analysis for Best Non Linear Model", expanded=True):
                st.header("📈 Learning Curve for Best Non Linear Model")

                def metrics_lcv_non_linear(df, x_train, x_test, y_train, y_test, model=best_model_non_linear):
                    """
                    Learning Curve for winner model
                    ------------
                    Parameters:
                    df: pd.DataFrame - Full dataframe
                    x_train: pd.DataFrame - Training features
                    x_test: pd.DataFrame - Testing features
                    y_train: pd.Series - Training target
                    y_test: pd.Series - Testing target
                    model: sklearn estimator - Fitted model to evaluate learning curve
                    ------------
                    Returns:
                    None - Displays learning curve plot
                    """
                    from sklearn.base import clone

                    train_sizes = np.linspace(10, len(x_train), 10).astype(int)
                    train_rmse_list = []
                    test_rmse_list  = []

                    for ts in train_sizes:
                        m = clone(model)
                        m.fit(x_train.iloc[:ts], y_train.iloc[:ts])
                        train_rmse_list.append(np.sqrt(mean_squared_error(y_train.iloc[:ts], m.predict(x_train.iloc[:ts]))))
                        test_rmse_list.append(np.sqrt(mean_squared_error(y_test, m.predict(x_test))))

                    train_mse_mean = np.array(train_rmse_list)
                    test_mse_mean  = np.array(test_rmse_list)
                    lc_df = pd.DataFrame({
                        "Train Size": train_sizes,
                        "Train Score": train_mse_mean,
                        "Test Score": test_mse_mean
                    })
                    lc_df["Gap"] = lc_df["Test Score"] - lc_df["Train Score"]

                    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
                    last_step = list(model.named_steps.values())[-1]
                    ax[0].plot(train_sizes, train_mse_mean, label="Train RMSE", color="steelblue")
                    ax[0].plot(train_sizes, test_mse_mean, label="Test RMSE (holdout)", color="orange")
                    ax[0].set_title(f"Learning Curve — {last_step.__class__.__name__}", fontweight="bold", fontsize=8, pad=15)
                    ax[0].set_xlabel("Training Set Size")
                    ax[0].set_ylabel("RMSE")
                    sns.despine(ax=ax[0])
                    ax[0].legend()
                    ax[0].grid()

                    sns.barplot(x=["Train", "Test"], y=[train_mse_mean.mean(), test_mse_mean.mean()], ax=ax[1], palette=["lightblue", "orange"])
                    ax[1].set_title(f"Average RMSE at Different Training Sizes", fontweight="bold", fontsize=8, pad=15)
                    ax[1].set_ylabel("RMSE")
                    sns.despine(ax=ax[1])
                    ax[1].grid(axis="y")

                    sns.lineplot(x=lc_df["Train Size"], y=lc_df["Gap"], marker="o", color="coral", label="RMSE gap Train - Test", ax=ax[2])
                    ax[2].set_title(f"Gap Between Train and Test RMSE ", fontweight="bold", fontsize=8, pad=15)
                    ax[2].set_xlabel("Training Set Size")
                    ax[2].set_ylabel(f"RMSE Score Gap (Train - Test)")
                    ax[2].annotate(
                        text=f"Min Gap =\n{lc_df['Gap'].min():.4f}",
                        xy=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()], lc_df["Gap"].min()),
                        xytext=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()] + 10, lc_df["Gap"].min() + 0.05),
                        textcoords="data",
                        arrowprops= dict(arrowstyle="->", color="black"),
                        fontsize=6, fontweight="bold"
                    )
                    ax[2].set_ylim(lc_df["Gap"].min() - 0.02, lc_df["Gap"].max() + 0.05)
                    sns.despine(ax=ax[2])
                    ax[2].legend(loc="upper right")

                    st.pyplot(fig)

                metrics_lcv_non_linear(df_model, train_lin[predictors], test_lin[predictors], train_lin["Score"], test_lin["Score"], model=best_model_non_linear)
            # --------------------------- EXPLANATORY POWER -----------------------------
            with st.expander("📊 Explanatory Power of Predictors", expanded=False):
                st.subheader("📊 Explanatory Power of Predictors")
                 # Background dataset — small sample is enough for speed
                X_background = shap.sample(train_lin[predictors], 100)

                explainer_non_linear   = shap.Explainer(best_model_non_linear.predict, X_background)
                shap_values_non_linear = explainer_non_linear(test_lin[predictors])
                col1, col2 = st.columns(2)
                with col1:
                    # Beeswarm
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.beeswarm(shap_values_non_linear, show=False)
                    st.pyplot(fig)
                with col2:
                    # Waterfall
                    sample_ind = 0
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values_non_linear[sample_ind], max_display=14, show=False)
                    st.pyplot(fig)
        elif models == "Bagging & Boosting Models":

            @st.cache_data(show_spinner="Training ensemble models... (runs once per dataset)")
            def fit_ensemble_models(X_train, y_train, X_test, y_test):

                """
                Fit RF, Adaboost and Gradient Boosting with grid search hyperparameters
                ------------
                Parameters:
                X_train: pd.DataFrame - Training features
                y_train: pd.Series - Training target
                X_test: pd.DataFrame - Testing features
                y_test: pd.Series - Testing target
                ------------
                Returns:
                dict - Dictionary of model name to fitted model and performance metrics
                """
                results = []
                #------------------ RF ---------------------------
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.ensemble import AdaBoostRegressor
                from sklearn.ensemble import GradientBoostingRegressor
                pipe_rf = Pipeline([
                    ("rf", RandomForestRegressor(criterion="squared_error", n_jobs=4))
                ])
                grid_rf = GridSearchCV(pipe_rf,
                                       param_grid=
                                       {"rf__n_estimators": [100, 200, 300],
                                        "rf__max_depth": [None, 10, 20, 30],
                                        "rf__min_samples_leaf": [2, 5, 10],
                                        "rf__max_features": ["sqrt", "log2", 0.33, 0.5]},
                                        cv=5, scoring="neg_mean_squared_error")
                grid_rf.fit(X_train, y_train)
                best_rf = grid_rf.best_estimator_
                y_pred_train_rf = best_rf.predict(X_train)
                y_pred_test_rf = best_rf.predict(X_test)

                # ------------------ AdaBoost ---------------------------
                param_grid = [
                    # DT — no scaling needed
                    {
                        "ada__estimator":                  [DecisionTreeRegressor()],
                        "ada__estimator__max_depth":       [1, 2, 3], #DT hyperparameter
                        "ada__estimator__min_samples_leaf":[1, 5, 10], #DT hyperparameter
                        "ada__n_estimators":               [100, 200, 300],
                        "ada__learning_rate":              [0.01, 0.1, 0.5],
                    },
                    # SVR — scaling required
                    {
                        "ada__estimator":                  [SVR()],
                        "ada__estimator__C":               [0.1, 1.0, 10.0],#SVR Hyperparameter
                        "ada__estimator__kernel":          ["rbf", "linear"], #SVR Hyperparameter
                        "ada__n_estimators":               [100, 200], #Adaboost
                        "ada__learning_rate":              [0.01, 0.1],#Adaboost
                    },
                ]

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ada", AdaBoostRegressor())
                ])

                grid_ada = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,   # note: now uses "ada__" prefix
                    cv=5,
                    scoring="neg_mean_squared_error",
                    n_jobs=4
                )
                grid_ada.fit(X_train, y_train)
                best_ada = grid_ada.best_estimator_
                y_pred_train_ada = best_ada.predict(X_train)
                y_pred_test_ada = best_ada.predict(X_test)

                # ------------------ Gradient Boosting ---------------------------
                pipe_gb = Pipeline([
                    ("scaler", StandardScaler()),
                    ("gb", GradientBoostingRegressor(loss="squared_error"))
                ])
                grid_gb = GridSearchCV(pipe_gb,
                                        param_grid={
                                            "gb__learning_rate": np.logspace(-3, 0, 10),
                                            "gb__n_estimators": [100, 200, 300],
                                            "gb__max_depth": [3, 4, 5],
                                            "gb__max_features": ["auto", "sqrt", "log2"],
                                            "gb__subsample": [0.6, 0.8, 1.0]
                                        },
                                        cv=5,
                                        scoring="neg_mean_squared_error",
                                        n_jobs=4
                )
                grid_gb.fit(X_train, y_train)
                best_gb = grid_gb.best_estimator_
                y_pred_train_gb = best_gb.predict(X_train)
                y_pred_test_gb = best_gb.predict(X_test)

                # ------------------- Compile results ---------------------------
                results.append({
                    "Model": f"Random Forest (n_estimators={grid_rf.best_params_['rf__n_estimators']}, max_depth={grid_rf.best_params_['rf__max_depth']}, min_samples_leaf={grid_rf.best_params_['rf__min_samples_leaf']}, max_features={grid_rf.best_params_['rf__max_features']})",
                    "Train R²": r2_score(y_train, y_pred_train_rf),
                    "Test R²": r2_score(y_test, y_pred_test_rf),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_rf),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_rf),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_rf)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_rf),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_rf)
                })
                results.append({
                    "Model": f"AdaBoost (estimator={grid_ada.best_params_['ada__estimator'].__class__.__name__}, n_estimators={grid_ada.best_params_['ada__n_estimators']}, learning_rate={grid_ada.best_params_['ada__learning_rate']})",
                    "Train R²": r2_score(y_train, y_pred_train_ada),
                    "Test R²": r2_score(y_test, y_pred_test_ada),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_ada),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_ada),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_ada)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_ada)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_ada),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_ada)
                })
                results.append({
                    "Model": f"Gradient Boosting (n_estimators={grid_gb.best_params_['gb__n_estimators']}, learning_rate={grid_gb.best_params_['gb__learning_rate']}, max_depth={grid_gb.best_params_['gb__max_depth']}, max_features={grid_gb.best_params_['gb__max_features']}, subsample={grid_gb.best_params_['gb__subsample']})",
                    "Train R²": r2_score(y_train, y_pred_train_gb),
                    "Test R²": r2_score(y_test, y_pred_test_gb),
                    "Train MSE": mean_squared_error(y_train, y_pred_train_gb),
                    "Test MSE": mean_squared_error(y_test, y_pred_test_gb),
                    "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train_gb)),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test_gb)),
                    "Train MAE": mean_absolute_error(y_train, y_pred_train_gb),
                    "Test MAE": mean_absolute_error(y_test, y_pred_test_gb)
                })
                model_objects = [best_rf, best_ada, best_gb]
                results_df = pd.DataFrame(results).sort_values(by="Test RMSE", ascending=True)
                best_model_obj = model_objects[results_df.index[0]]

                return best_rf, best_ada, best_gb, results_df, best_model_obj

            H= 40    #Test size of 40 samples
            train_lin = df_model.iloc[:-H].copy()
            test_lin  = df_model.iloc[-H:].copy()
            #train_lin, test_lin = train_test_split(df_model, test_size=0.2, shuffle=False, stratify=None)
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]

            time_start = time.time()
            rf, ada, gb, results, best_model_ensemble = fit_ensemble_models(train_lin[predictors], train_lin["Score"], test_lin[predictors], test_lin["Score"])
            time_end = time.time()
            st.header(f"📊 Results Dataframe")
            st.dataframe(results)
            st.badge(f"Best model: {results.iloc[0]['Model']}", color="green", icon="✅")
            st.badge(f"Best parameters: {best_model_ensemble.get_params()}", icon="✅")
            st.info(f"Time taken to fit models: {time_end - time_start:.2f} seconds")

            # ----------------------------- BEST MODEL PERFORMANCE -----------------------------
            st.header(f"📊 Performance of Ensemble Models")
            y_pred_train_best_ensemble = best_model_ensemble.predict(train_lin[predictors])
            y_pred_test_best_ensemble = best_model_ensemble.predict(test_lin[predictors])
            r2_train_ensemble = r2_score(train_lin["Score"], y_pred_train_best_ensemble)
            r2_test_ensemble = r2_score(test_lin["Score"], y_pred_test_best_ensemble)
            rmse_train_ensemble = np.sqrt(mean_squared_error(train_lin["Score"], y_pred_train_best_ensemble))
            rmse_test_ensemble = np.sqrt(mean_squared_error(test_lin["Score"], y_pred_test_best_ensemble))
            mae_train_ensemble = mean_absolute_error(train_lin["Score"], y_pred_train_best_ensemble)
            mae_test_ensemble = mean_absolute_error(test_lin["Score"], y_pred_test_best_ensemble)
            mse_train_ensemble = rmse_train_ensemble ** 2
            mse_test_ensemble = rmse_test_ensemble ** 2

    #----------------------------- PERFORMANCE METRICS -----------------------------
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
                    st.metric("Test R²", f"{r2_test_ensemble:.3f}", delta=f"{r2_test_ensemble - r2_train_ensemble:.3f}",
                            delta_color="green" if r2_test_ensemble > r2_train_ensemble else "red")
                with c2:
                    st.metric("Test MSE", f"{mse_test_ensemble:.3f}", delta=f"{mse_test_ensemble - mse_train_ensemble:.3f}",
                                    delta_color="red" if mse_test_ensemble > mse_train_ensemble else "green",
                                    help="Mean Squared Error (MSE): lower values indicate better fit.\
                                        Penalizes larger errors more heavily.")
                with c3:
                    st.metric("Test MAE", f"{mae_test_ensemble:.3f}", delta=f"{mae_test_ensemble - mae_train_ensemble:.3f}",
                                    delta_color="red" if mae_test_ensemble > mae_train_ensemble else "green",
                                    help="Mean Absolute Error (MAE): lower values indicate better fit.")
                with c4:
                    st.metric("Test RMSE", f"{rmse_test_ensemble:.3f}", delta=f"{rmse_test_ensemble - rmse_train_ensemble:.3f}",
                                    delta_color="red" if rmse_test_ensemble > rmse_train_ensemble else "green",
                                    help="Root Mean Squared Error (RMSE): lower values indicate better fit, in original units.")
                with c5:
                    st.metric("Test Samples", f"{test_lin.shape[0]}", help="The last 40 samples used for testing.")
                with c6:
                    st.metric("Test Start Date", f"{test_lin.Date.min().date()}")

            # ----------------------------- LEARNING CURVE -----------------------------
            with st.expander("📈 Learning Curve Analysis for Best Ensemble Model", expanded=True):
                st.header("📈 Learning Curve for Best Ensemble Model")

                def metrics_lcv_ensemble(df, x_train, x_test, y_train, y_test, model=best_model_ensemble):
                    """
                    Learning Curve for winner model
                    ------------
                    Parameters:
                    df: pd.DataFrame - Full dataframe
                    x_train: pd.DataFrame - Training features
                    x_test: pd.DataFrame - Testing features
                    y_train: pd.Series - Training target
                    y_test: pd.Series - Testing target
                    model: sklearn estimator - Fitted model to evaluate learning curve
                    ------------
                    Returns:
                    None - Displays learning curve plot
                    """
                    from sklearn.base import clone

                    train_sizes = np.linspace(10, len(x_train), 10).astype(int)
                    train_rmse_list = []
                    test_rmse_list  = []

                    for ts in train_sizes:
                        m = clone(model)
                        m.fit(x_train.iloc[:ts], y_train.iloc[:ts])
                        train_rmse_list.append(np.sqrt(mean_squared_error(y_train.iloc[:ts], m.predict(x_train.iloc[:ts]))))
                        test_rmse_list.append(np.sqrt(mean_squared_error(y_test, m.predict(x_test))))

                    train_mse_mean = np.array(train_rmse_list)
                    test_mse_mean  = np.array(test_rmse_list)
                    lc_df = pd.DataFrame({
                        "Train Size": train_sizes,
                        "Train Score": train_mse_mean,
                        "Test Score": test_mse_mean
                    })
                    lc_df["Gap"] = lc_df["Test Score"] - lc_df["Train Score"]

                    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
                    last_step = list(model.named_steps.values())[-1]
                    ax[0].plot(train_sizes, train_mse_mean, label="Train RMSE", color="steelblue")
                    ax[0].plot(train_sizes, test_mse_mean, label="Test RMSE (holdout)", color="orange")
                    ax[0].set_title(f"Learning Curve — {last_step.__class__.__name__}", fontweight="bold", fontsize=8, pad=15)
                    ax[0].set_xlabel("Training Set Size")
                    ax[0].set_ylabel("RMSE")
                    sns.despine(ax=ax[0])
                    ax[0].legend()
                    ax[0].grid()

                    sns.barplot(x=["Train", "Test"], y=[train_mse_mean.mean(), test_mse_mean.mean()], ax=ax[1], palette=["lightblue", "orange"])
                    ax[1].set_title(f"Average RMSE at Different Training Sizes", fontweight="bold", fontsize=8, pad=15)
                    ax[1].set_ylabel("RMSE")
                    sns.despine(ax=ax[1])
                    ax[1].grid(axis="y")

                    sns.lineplot(x=lc_df["Train Size"], y=lc_df["Gap"], marker="o", color="coral", label="RMSE gap Train - Test", ax=ax[2])
                    ax[2].set_title(f"Gap Between Train and Test RMSE ", fontweight="bold", fontsize=8, pad=15)
                    ax[2].set_xlabel("Training Set Size")
                    ax[2].set_ylabel(f"RMSE Score Gap (Train - Test)")
                    ax[2].annotate(
                        text=f"Min Gap =\n{lc_df['Gap'].min():.4f}",
                        xy=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()], lc_df["Gap"].min()),
                        xytext=(lc_df["Train Size"].iloc[lc_df["Gap"].idxmin()] + 10, lc_df["Gap"].min() + 0.05),
                        textcoords="data",
                        arrowprops= dict(arrowstyle="->", color="black"),
                        fontsize=6, fontweight="bold"
                    )
                    ax[2].set_ylim(lc_df["Gap"].min() - 0.02, lc_df["Gap"].max() + 0.05)
                    sns.despine(ax=ax[2])
                    ax[2].legend(loc="upper right")

                    st.pyplot(fig)

                metrics_lcv_ensemble(df_model, train_lin[predictors], test_lin[predictors], train_lin["Score"], test_lin["Score"], model=best_model_ensemble)
            # --------------------------- EXPLANATORY POWER -----------------------------
            with st.expander("📊 Explanatory Power of Predictors", expanded=False):
                st.subheader("📊 Explanatory Power of Predictors")
                 # Background dataset — small sample is enough for speed
                X_background = shap.sample(train_lin[predictors], 100)

                explainer_ensemble   = shap.Explainer(best_model_ensemble.predict, X_background)
                shap_values_ensemble = explainer_ensemble(test_lin[predictors])
                col1, col2 = st.columns(2)
                with col1:
                    # Beeswarm
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.beeswarm(shap_values_ensemble, show=False)
                    st.pyplot(fig)
                with col2:
                    # Waterfall
                    sample_ind = 0
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values_ensemble[sample_ind], max_display=14, show=False)
                    st.pyplot(fig)

    # ---------------------------------- UNSUPERVISED LEARNING ----------------------------------
    elif types == "Unsupervised":
        from sklearn.decomposition import PCA
        unsupervised_options = ["PCA", "T-SNE", "K-Means", "DBSCAN"]
        unsupervised_choice = st.selectbox("Select unsupervised technique:", options=unsupervised_options, key="unsupervised_choice")
        if unsupervised_choice == "PCA":
            st.header("📊 Principal Component Analysis (PCA)")
            df_model["Bad_Sleep"] = (df_model["Score"] < 80).astype(int)
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]
            X_scaled = StandardScaler().fit_transform(df_model[predictors])
            best_pca = PCA(n_components=0.85)
            X_pca = best_pca.fit_transform(X_scaled)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], ax=ax, hue=df_model["Bad_Sleep"], palette=["green", "red"], alpha=0.7)
            ax.set_title("PCA: First Two Principal Components")
            sns.despine(ax=ax)
            st.pyplot(fig)

            num_components =  [1, 2, 3, 4, 5, 6, 7]
            results = []
            for i in num_components:
                pca = PCA(n_components=i, random_state=42)
                pca_results = pca.fit_transform(X_scaled)
                explained_variance = pca.explained_variance_ratio_.sum() * 100
                explained_var_ = pca.explained_variance_ratio_
                results.append({
                    "Components": i,
                    "Explained Variance (%)": explained_variance,
                    "Explained Variance": explained_var_
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            # PVE Proportion of Variance Explained

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            results_df.plot(kind="bar", x="Components", y="Explained Variance (%)", color=sns.color_palette("icefire", n_colors=len(results_df)), legend=False, ax=ax[1])
            bars = plt.gca().patches
            values = results_df["Explained Variance (%)"].values
            for bar, value in zip(bars, values):
                ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax[1].set_title("Explained Variance by Number of PCA Components", fontsize=10, fontweight="bold", pad=15)
            ax[1].set_xlabel("Number of Components")
            ax[1].set_ylabel("Explained Variance (%)")
            ax[1].set_ylim(0, 100)
            sns.despine(ax=ax[1])
            plt.tight_layout()

            pca_df = pd.DataFrame(best_pca.components_, columns=predictors, index=[f"PC{i+1}" for i in range(best_pca.n_components_)])
            sns.heatmap(pca_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                        cbar_kws={"label": "Loading"}, vmin=-1, vmax=1, ax=ax[0])
            ax[0].set_title("PCA Loadings — each cell shows feature contribution to each PC",
                    fontweight="bold", fontsize=10, pad=15)
            plt.tight_layout()
            st.pyplot(fig)
        elif unsupervised_choice == "T-SNE":
            st.header("📊 T-Distributed Stochastic Neighbor Embedding (T-SNE)")
            from sklearn.manifold import TSNE
            df_model["Bad_Sleep"] = (df_model["Score"] < 80).astype(int)
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]
            X_scaled = StandardScaler().fit_transform(df_model[predictors])
            tsne = TSNE(n_components=2, perplexity=10)
            X_tsne = tsne.fit_transform(X_scaled)
            df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], ax=ax, hue=df_model["Bad_Sleep"].map({0:"Good Sleep", 1:"Bad Sleep"}), palette=["green", "red"], alpha=0.7)
            ax.set_title("T-SNE: 2D Visualization of Sleep Data")
            sns.despine(ax=ax)
            st.pyplot(fig)

        elif unsupervised_choice == "K-Means":
            st.header("📊 K-Means")
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]
            df_model["Bad_Sleep"] = (df_model["Score"] < 80).astype(int)
            X_scaled = StandardScaler().fit_transform(df_model[predictors])


            # Train K-Means for K=2 to 10 and evaluate with Inertia, Silhouette Score, and Davies-Bouldin Score
            results_cluster = []
            for k in range(2, 11): # K from 2 to 10
                kmeans = KMeans(n_clusters=k, random_state=42, algorithm="elkan")
                kmeans.fit(X_scaled)
                results_cluster.append({
                    "K": k,
                    "Inertia": kmeans.inertia_,
                    "Silhouette Score": silhouette_score(X_scaled, kmeans.labels_),
                    "Davies-Bouldin Score": davies_bouldin_score(X_scaled, kmeans.labels_)
                })
            results_cluster_df = pd.DataFrame(results_cluster).sort_values(by="Silhouette Score", ascending=False)
            st.dataframe(results_cluster_df)


            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            plt.suptitle("K-Means Clustering Evaluation Metrics vs K", fontsize=16, fontweight="bold")
            sns.lineplot(x="K", y="Inertia", data=results_cluster_df, marker="o", color="blue", ax=axes[0])
            axes[0].set_title("K-Means Inertia vs K", fontsize=8, fontweight="bold")
            axes[0].set_xlabel("Number of Clusters (K)")
            axes[0].set_ylabel("Inertia")
            axes[0].set_xticks(results_cluster_df["K"])
            axes[0].grid()
            sns.despine(ax=axes[0])

            sns.lineplot(x="K", y="Silhouette Score", data=results_cluster_df, marker="o", color="orange", ax=axes[1])
            axes[1].set_title("K-Means Silhouette Score vs K", fontsize=8, fontweight="bold")
            axes[1].set_xlabel("Number of Clusters (K)")
            axes[1].set_ylabel("Silhouette Score")
            axes[1].set_xticks(results_cluster_df["K"])
            axes[1].grid()
            sns.despine(ax=axes[1])

            sns.lineplot(x="K", y="Davies-Bouldin Score", data=results_cluster_df, marker="o", color="green", ax=axes[2])
            axes[2].set_title("K-Means Davies-Bouldin Score vs K", fontsize=8, fontweight="bold")
            axes[2].set_xlabel("Number of Clusters (K)")
            axes[2].set_ylabel("Davies-Bouldin Score")
            axes[2].set_xticks(results_cluster_df["K"])
            axes[2].grid()
            sns.despine(ax=axes[2])

            plt.tight_layout()
            st.pyplot(fig)

            # --------------------------------------
            # Select best K and fit the K-Means model
            best_k = results_cluster_df.iloc[0]["K"]
            kmeans = KMeans(n_clusters=int(best_k), random_state=42, algorithm="elkan")
            kmeans.fit(X_scaled)
            df_model["KMeans_Cluster"] = kmeans.labels_

            st.subheader(f"📊 Cluster Analysis for K={int(best_k)}")
            cluster_analysis = df_model.groupby("KMeans_Cluster").agg({
                "Score": "mean",
                "REM hrs": "mean",
                "Stress_prev_day": "mean",
                "Deep hrs": "mean",
                "Wake Count": "mean",
                "Sleep_hr_surplus": "mean",
                "Respiration": "mean",
                "Stress_sleep": "mean"
            }).round(2)

            st.dataframe(cluster_analysis)

            st.subheader("📊 T-Distributed Stochastic Neighbor Embedding (T-SNE)")
            from sklearn.manifold import TSNE
            predictors = ["REM hrs", "Stress_prev_day", "Deep hrs", "Wake Count", "Sleep_hr_surplus", "Respiration", "Stress_sleep"]
            X_scaled = StandardScaler().fit_transform(df_model[predictors])
            tsne = TSNE(n_components=2, perplexity=10)
            X_tsne = tsne.fit_transform(X_scaled)
            df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], ax=ax, hue=df_model["KMeans_Cluster"], palette="viridis", alpha=0.7)
            ax.set_title("T-SNE: 2D Visualization of Sleep Data")
            sns.despine(ax=ax)
            st.pyplot(fig)

st.caption(
    "Tip: If you only train 3–4 days/week, use weekly aggregation (Volume / mean Recovery / mean Sleep) "
    "to avoid the mismatch between daily sleep and training frequency."
)
