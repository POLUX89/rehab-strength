"""Manual synthetic data generator (demo mode).

python -m rehab_strength.synthetic.generate  ->  data/synthetic/*.csv

Design: a daily "wellness" latent state with AR(1) structure induces both the
temporal autocorrelation (the TSA tab shows real ACF, not white noise) and the
cross-column correlations (more stress -> worse score, more sleep -> better
recovery). Derived columns (Efficiency, z-scores, Sigmoid, naps) are NOT
synthesized: they are recomputed with the ingestion pipeline's own
formulas/functions. Generic plausible parameters -- nothing is fitted to the
real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..ingest.sleep import RECOVERY_COMPONENTS, classify_nap, nap_duration, nap_status

SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
END_DATE = "2026-07-20"  # last date of the dataset (fixed -> reproducible CSVs)
N_DAYS = 365

# Exact columns the app expects (same order as the real CSVs).
SLEEP_COLUMNS = [
    "Date",
    "Main_x",
    "Start",
    "End",
    "InBed hrs",
    "Asleep hrs",
    "Awake",
    "REM hrs",
    "Light hrs",
    "Deep hrs",
    "Wake Count",
    "Efficiency",
    "Fall Asleep",
    "Data Source",
    "Main_y",
    "Start_Nap",
    "End_Nap",
    "Asleep_Nap",
    "Data Source Nap",
    "Overnight HRV",
    "Baseline",
    "7d Avg",
    "Stress",
    "RHR",
    "Stress_sleep",
]
RECOVERY_EXTRA_COLUMNS = [
    "Score",
    "Resting Heart Rate",
    "Body Battery",
    "Pulse Ox",
    "Respiration",
    "HRV Status",
    "Quality",
    "Duration",
    "Sleep Need",
    "Bedtime",
    "Wake Time",
    "Stress_prev_day",
    "Z Asleep hrs",
    "Z Overnight HRV",
    "Z Resting Heart Rate",
    "Z Score",
    "Z Stress_prev_day",
    "RECOVERY_SCORE_RAW",
    "Sigmoid Recovery Score",
    "End_Nap_Decimal",
    "Start_Nap_Decimal",
    "Nap_Classifier",
    "Nap_Duration_Score",
    "Nap Status",
    "Sigmoid with Nap",
    "DELTA_NAP",
]
WORKOUT_COLUMNS = [
    "DATE",
    "WORKOUT_NAME",
    "DURATION_MIN",
    "EXERCISE_NAME",
    "SET_ORDER",
    "WEIGHT_LBS",
    "REPS",
    "NOTES",
    "RPE",
    "VOLUME",
    "RECORDED_RPE",
    "REPS_ONLY",
]

# Generic training split: weekday -> (name, exercises).
# Base weights in lbs; smooth linear progression + per-session noise.
_TRAINING_SPLIT = {
    0: (
        "PUSH DAY",
        [
            ("Bench Press (Barbell)", 135),
            ("Overhead Press (Barbell)", 75),
            ("Incline Press (Dumbbell)", 50),
            ("Lateral Raise (Dumbbell)", 20),
            ("Triceps Pushdown (Cable)", 45),
        ],
    ),
    1: (
        "LEGS DAY",
        [
            ("Squat (Barbell)", 155),
            ("Romanian Deadlift (Barbell)", 135),
            ("Leg Press (Machine)", 250),
            ("Leg Curl (Machine)", 80),
            ("Calf Raise (Machine)", 120),
        ],
    ),
    3: (
        "PULL DAY",
        [
            ("Deadlift (Barbell)", 185),
            ("Lat Pulldown (Machine)", 110),
            ("Seated Row (Cable)", 100),
            ("Bicep Curl (Dumbbell)", 30),
            ("Face Pull (Cable)", 40),
        ],
    ),
    5: (
        "FULL BODY",
        [
            ("Squat (Barbell)", 145),
            ("Bench Press (Barbell)", 125),
            ("Seated Row (Cable)", 95),
            ("Overhead Press (Dumbbell)", 35),
            ("Plank Hold", 25),
        ],
    ),
}
_WORKOUT_NOTES = [
    "Felt strong today",
    "Slept badly, lighter loads",
    "New PR!",
    "Short on time",
    "Great pump",
]


def _fmt_h_min(hours: float) -> str:
    """Format decimal hours as the app's ``"7h 30min"`` string."""
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h}h {m}min"


def _fmt_hhmm(minutes_from_midnight: float) -> str:
    """Format minutes-from-midnight (may exceed 24 h) as ``"HH:MM"``."""
    total = int(round(minutes_from_midnight)) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _latent_ar1(n: int, rng: np.random.Generator, phi: float = 0.7) -> np.ndarray:
    """Standardized AR(1) latent state driving temporal autocorrelation.

    Args:
        n: Number of days.
        rng: Seeded generator.
        phi: AR(1) coefficient (persistence of good/bad streaks).

    Returns:
        Array of length ``n`` with ~N(0, 1) marginals and lag-1
        autocorrelation ≈ ``phi``.
    """
    z = np.empty(n)
    z[0] = rng.normal()
    innovation_sd = np.sqrt(1 - phi**2)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal(scale=innovation_sd)
    return z


def generate_recovery(
    n_days: int = N_DAYS, end_date: str = END_DATE, seed: int = 42
) -> pd.DataFrame:
    """Generate the synthetic ``clean_recovery_data`` frame (51 columns).

    A daily AR(1) wellness state ``w`` induces both autocorrelation and
    cross-column correlations (stress ↓, HRV ↑, sleep ↑, score ↑ when ``w`` is
    high). Derived columns (Efficiency, z-scores, sigmoid, nap scoring) are
    recomputed with the ingestion pipeline's own formulas, never sampled.

    Args:
        n_days: Number of consecutive daily rows.
        end_date: Last date (ISO) — fixed by default so the CSV is reproducible.
        seed: RNG seed.

    Returns:
        DataFrame with the exact real-schema columns, formats included.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end_date, periods=n_days, freq="D")
    w = _latent_ar1(n_days, rng)

    # --- physiological roots (real units, generic plausible ranges) ---
    asleep = np.clip(7.1 + 0.55 * w + rng.normal(0, 0.45, n_days), 4.5, 9.8)
    rem = np.clip(asleep * (0.21 + 0.02 * w) + rng.normal(0, 0.15, n_days), 0.4, 3.0)
    deep = np.clip(asleep * (0.17 + 0.015 * w) + rng.normal(0, 0.12, n_days), 0.3, 2.5)
    light = np.clip(asleep - rem - deep, 0.5, None)
    awake_min = np.clip(35 - 12 * w + rng.normal(0, 12, n_days), 5, 130).round()
    in_bed = asleep + awake_min / 60.0
    wake_count = rng.poisson(np.clip(1.6 - 0.6 * w, 0.2, 4.0))
    hrv = np.clip(52 + 9 * w + rng.normal(0, 4, n_days), 25, 90).round()
    rhr = np.clip(53 - 4 * w + rng.normal(0, 2.5, n_days), 40, 75).round()
    stress = np.clip(26 - 7 * w + rng.normal(0, 5, n_days), 3, 80).round()
    stress_sleep = np.clip(14 - 4 * w + rng.normal(0, 4, n_days), 1, 60).round()
    respiration = np.clip(14.4 + 0.3 * w + rng.normal(0, 0.7, n_days), 11, 18).round(2)
    # Baseline calibrated for ~25% of days with Score < 80 (the minority
    # "Bad Sleep" class, as in a realistic log) without fitting the real data.
    score = np.clip(88 + 9 * w - 1.2 * wake_count + rng.normal(0, 5, n_days), 40, 100).round()
    body_battery = np.clip(45 + 20 * w + rng.normal(0, 8, n_days), 5, 100).round()
    pulse_ox = np.clip(95 + rng.normal(0, 1.1, n_days), 90, 100).round(2)

    # --- schedule ---
    bed_min = 23 * 60 + rng.normal(0, 35, n_days)  # ~23:00 +/- 35 min
    start = np.array([_fmt_hhmm(m) for m in bed_min])
    end = np.array([_fmt_hhmm(m + ib * 60) for m, ib in zip(bed_min, in_bed, strict=True)])
    fall_asleep = np.clip(rng.exponential(18, n_days), 0, 110).round()

    # --- naps (~25% of days) ---
    has_nap = rng.random(n_days) < 0.25
    nap_start_min = 13.5 * 60 + rng.normal(0, 80, n_days)
    nap_len = np.clip(rng.normal(32, 14, n_days), 12, 90).round()
    start_nap = np.where(has_nap, [_fmt_hhmm(m) for m in nap_start_min], None)
    end_nap = np.where(
        has_nap,
        [_fmt_hhmm(m + length) for m, length in zip(nap_start_min, nap_len, strict=True)],
        None,
    )
    asleep_nap = np.where(has_nap, nap_len, np.nan)

    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Main_x": True,
            "Start": start,
            "End": end,
            "InBed hrs": in_bed.round(2),
            "Asleep hrs": asleep.round(2),
            "Awake": awake_min.astype(int),
            "REM hrs": rem.round(2),
            "Light hrs": light.round(2),
            "Deep hrs": deep.round(2),
            "Wake Count": wake_count.astype(float),
            "Efficiency": [f"{v:.0f}%" for v in np.clip(100 * asleep / in_bed, 55, 99)],
            "Fall Asleep": fall_asleep,
            "Data Source": "Synthetic",
            "Main_y": np.where(has_nap, "False", None),
            "Start_Nap": start_nap,
            "End_Nap": end_nap,
            "Asleep_Nap": asleep_nap,
            "Data Source Nap": np.where(has_nap, "Synthetic", None),
            "Overnight HRV": hrv,
            "Baseline": [f"{int(b - 5)}ms - {int(b + 5)}ms" for b in hrv],
            "7d Avg": pd.Series(hrv).rolling(7, min_periods=1).mean().round(),
            "Stress": stress,
            "RHR": rhr,
            "Stress_sleep": stress_sleep,
            "Score": score,
            "Resting Heart Rate": rhr,
            "Body Battery": body_battery,
            "Pulse Ox": pulse_ox,
            "Respiration": respiration,
            "HRV Status": hrv,
            "Quality": pd.cut(
                score,
                bins=[0, 60, 80, 90, 101],
                right=False,
                labels=["Poor", "Fair", "Good", "Excellent"],
            ).astype(str),
            "Duration": [_fmt_h_min(a) for a in asleep],
            "Sleep Need": [
                _fmt_h_min(v)
                for v in rng.choice([7.5, 7 + 40 / 60, 8.0], n_days, p=[0.6, 0.25, 0.15])
            ],
            "Bedtime": [f"{s}:00" for s in start],
            "Wake Time": [f"{e}:00" for e in end],
        }
    )

    # --- derived: same formulas as the pipeline (build_recovery) ---
    df["Stress_prev_day"] = df["Stress"].shift(1)
    for col in RECOVERY_COMPONENTS:
        df["Z " + col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    df["Z Resting Heart Rate"] = -df["Z Resting Heart Rate"]
    df["Z Stress_prev_day"] = -df["Z Stress_prev_day"]
    df["RECOVERY_SCORE_RAW"] = (
        df[["Z " + c for c in RECOVERY_COMPONENTS]].sum(axis=1, skipna=False) / 5
    )
    df["Sigmoid Recovery Score"] = 1 / (1 + np.exp(-df["RECOVERY_SCORE_RAW"]))

    end_nap_dt = pd.to_datetime(df["End_Nap"], format="%H:%M", errors="coerce")
    df["End_Nap_Decimal"] = end_nap_dt.dt.hour + end_nap_dt.dt.minute / 60
    start_nap_dt = pd.to_datetime(df["Start_Nap"], format="%H:%M", errors="coerce")
    df["Start_Nap_Decimal"] = start_nap_dt.dt.hour + start_nap_dt.dt.minute / 60
    df["Nap_Classifier"] = df["Start_Nap_Decimal"].apply(classify_nap)
    df["Nap_Duration_Score"] = df["Asleep_Nap"].apply(nap_duration)
    df["Nap Status"] = nap_status(df["Nap_Classifier"], df["Nap_Duration_Score"])
    df["Sigmoid with Nap"] = (
        df["Sigmoid Recovery Score"] + df["Nap_Classifier"] + df["Nap_Duration_Score"]
    ).clip(0, 1)
    df["DELTA_NAP"] = df["Sigmoid with Nap"] - df["Sigmoid Recovery Score"]

    return df[SLEEP_COLUMNS + RECOVERY_EXTRA_COLUMNS]


def sleep_from_recovery(recovery: pd.DataFrame) -> pd.DataFrame:
    """Slice the recovery frame down to the sleep CSV's 25 columns."""
    return recovery[SLEEP_COLUMNS].copy()


def generate_workouts(
    n_days: int = N_DAYS, end_date: str = END_DATE, seed: int = 42
) -> pd.DataFrame:
    """Simulate a hierarchical Strong export (sessions → exercises → sets).

    Rule-based, not learned: a fixed weekly split, linear progression with
    noise, occasional deloads, and RPE recorded on most (not all) sets —
    mirroring the structure of a real training log.

    Args:
        n_days: Calendar span (sessions land on the split's weekdays).
        end_date: Last calendar date (ISO).
        seed: RNG seed.

    Returns:
        DataFrame with the 12 columns of ``clean_strong_workouts.csv``.
    """
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range(end=end_date, periods=n_days, freq="D")
    rows = []
    for day_idx, date in enumerate(dates):
        split = _TRAINING_SPLIT.get(date.weekday())
        if split is None or rng.random() < 0.12:  # ~12% of sessions skipped
            continue
        workout_name, exercises = split
        week = day_idx / 7.0
        deload = rng.random() < 0.08
        start_time = date + pd.Timedelta(
            hours=int(rng.integers(7, 10)), minutes=int(rng.integers(0, 60))
        )
        stamp = start_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = int(np.clip(rng.normal(95, 15), 55, 140))
        note_session = rng.random() < 0.1
        for ex_name, base_w in exercises:
            n_sets = int(rng.integers(3, 5))
            progression = base_w * (1 + 0.004 * week) * (0.85 if deload else 1.0)
            weight = max(5.0, round(progression / 5) * 5 + float(rng.choice([-5, 0, 0, 5])))
            for set_order in range(1, n_sets + 1):
                reps = float(int(np.clip(rng.normal(9 - set_order * 0.7, 1.5), 4, 15)))
                recorded = bool(rng.random() < 0.8)
                rpe = (
                    float(np.clip(round(rng.normal(7.5 + set_order * 0.4)), 6, 10))
                    if recorded
                    else np.nan
                )
                rows.append(
                    {
                        "DATE": stamp,
                        "WORKOUT_NAME": workout_name,
                        "DURATION_MIN": duration,
                        "EXERCISE_NAME": ex_name,
                        "SET_ORDER": str(set_order),
                        "WEIGHT_LBS": weight,
                        "REPS": reps,
                        "NOTES": rng.choice(_WORKOUT_NOTES)
                        if (note_session and set_order == 1)
                        else "",
                        "RPE": rpe,
                        "VOLUME": weight * reps,
                        "RECORDED_RPE": recorded,
                        "REPS_ONLY": reps,
                    }
                )
    return pd.DataFrame(rows, columns=WORKOUT_COLUMNS)


def write_all(out_dir: Path = SYNTHETIC_DIR, seed: int = 42) -> tuple[Path, Path, Path]:
    """Generate and write the three synthetic CSVs the app consumes.

    Args:
        out_dir: Destination directory (created if missing).
        seed: RNG seed shared by the generators.

    Returns:
        The three written paths (workouts, sleep, recovery).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    recovery = generate_recovery(seed=seed)
    sleep = sleep_from_recovery(recovery)
    workouts = generate_workouts(seed=seed)

    paths = (
        out_dir / "clean_strong_workouts.csv",
        out_dir / "clean_sleep_data.csv",
        out_dir / "clean_recovery_data.csv",
    )
    workouts.to_csv(paths[0], index=False)
    sleep.to_csv(paths[1], index=False)
    recovery.to_csv(paths[2], index=False)
    for p, df in zip(paths, (workouts, sleep, recovery), strict=True):
        print(f"💾 {p.name}: {len(df)} rows, {len(df.columns)} columns")
    return paths


if __name__ == "__main__":
    write_all()
