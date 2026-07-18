"""Helpers de transformación de datos puros (sin Streamlit).

Extraídos de streamlit_app.py sin cambiar la lógica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_unique_columns(cols):
    """Fix duplicate column names by suffixing .1 .2 ... (Streamlit uploads can keep dupes)."""
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
        w = float(w)
        r = float(r)
        if np.isnan(w) or np.isnan(r):
            return np.nan
        return w * (1.0 + r / 30.0)
    except Exception:
        return np.nan


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def daily_ma(series, window_days):
    # series is indexed by datetime daily (or resampled)
    return series.rolling(window=window_days, min_periods=max(1, window_days // 2)).mean()


def weekly_bucket(dt_series):
    dt = pd.to_datetime(dt_series, errors="coerce")  # coerce bad strings to NaT
    return dt.dt.to_period("W-MON").dt.start_time


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
    if x >= 0.7:
        return "🟢 ⬆️ Ready"
    if x >= 0.55:
        return "🟡 Moderate"
    return "🔴 ⬇️ Low"


def sleep_classifier(q):
    return 1 if q in ["Good", "Excellent"] else 0


def string_to_decimal_hours(time_str):
    if pd.isna(time_str):
        return np.nan
    time_str = time_str.strip()
    if "h" in time_str and "min" in time_str:
        hours, minutes = time_str.split("h")
        minutes = minutes.replace("min", "").strip()
        return float(hours.strip()) + float(minutes) / 60
    elif "h" in time_str:
        hours = time_str.replace("h", "").strip()
        return float(hours)
    elif "min" in time_str:
        minutes = time_str.replace("min", "").strip()
        return float(minutes) / 60
    else:
        return np.nan


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
    for cand in [
        "Score",
        "Wake Count",
        "Efficiency",
        "Asleep hrs",
        "InBed hrs",
        "REM hrs",
        "Light hrs",
        "Deep hrs",
    ]:
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
    for cand in [
        "Sigmoid Recovery Score",
        "RECOVERY_SCORE_RAW",
        "Stress_prev_day",
        "Overnight HRV",
        "Resting Heart Rate",
        "Score",
    ]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df
