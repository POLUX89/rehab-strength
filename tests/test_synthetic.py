"""Tests for the synthetic data generator and its contract with the app.

Two layers: (1) the in-memory generator -- exact schema, formats, physical
constraints and per-seed determinism; (2) the COMMITTED CSVs in data/synthetic/
-- that they exist, pass through normalize_* and survive the Models tab path
(including the classification guard, which requires >=50 rows).
"""

from pathlib import Path

import pandas as pd
import pytest

from app.helpers.transforms import (
    normalize_recovery,
    normalize_sleep,
    normalize_workouts,
    string_to_decimal_hours,
)
from rehab_strength.synthetic.generate import (
    RECOVERY_EXTRA_COLUMNS,
    SLEEP_COLUMNS,
    WORKOUT_COLUMNS,
    generate_recovery,
    generate_workouts,
    sleep_from_recovery,
)

SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"

MODEL_PREDICTORS = [
    "REM hrs",
    "Stress_prev_day",
    "Deep hrs",
    "Wake Count",
    "Sleep_hr_surplus",
    "Respiration",
    "Stress_sleep",
]


# ---------------------------------------------------------------------------
# Capa 1: generador en memoria
# ---------------------------------------------------------------------------
def test_recovery_schema_and_formats():
    rec = generate_recovery(n_days=90, seed=7)
    assert list(rec.columns) == SLEEP_COLUMNS + RECOVERY_EXTRA_COLUMNS
    # formatos que la app parsea
    assert rec["Efficiency"].str.endswith("%").all()
    assert rec["Sleep Need"].str.match(r"^\d+h \d+min$").all()
    assert rec["Duration"].str.match(r"^\d+h \d+min$").all()
    assert rec["Start"].str.match(r"^\d{2}:\d{2}$").all()
    # fechas diarias consecutivas
    dates = pd.to_datetime(rec["Date"])
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()


def test_recovery_physical_constraints():
    rec = generate_recovery(n_days=120, seed=3)
    assert (rec["InBed hrs"] >= rec["Asleep hrs"]).all()
    stages = rec[["REM hrs", "Light hrs", "Deep hrs"]].sum(axis=1)
    assert ((stages - rec["Asleep hrs"]).abs() < 1.0).all()  # stages ~= total
    assert rec["Score"].between(0, 100).all()
    assert (rec["Wake Count"] >= 0).all()
    # the recomputed sigmoid lives in (0, 1); first row is NaN due to the shift
    sig = rec["Sigmoid Recovery Score"].dropna()
    assert sig.between(0, 1).all()
    assert pd.isna(rec["RECOVERY_SCORE_RAW"].iloc[0])


def test_generator_is_deterministic():
    a = generate_recovery(n_days=60, seed=42)
    b = generate_recovery(n_days=60, seed=42)
    pd.testing.assert_frame_equal(a, b)
    wa = generate_workouts(n_days=60, seed=42)
    wb = generate_workouts(n_days=60, seed=42)
    pd.testing.assert_frame_equal(wa, wb)


def test_sleep_is_recovery_prefix():
    rec = generate_recovery(n_days=30, seed=1)
    sl = sleep_from_recovery(rec)
    assert list(sl.columns) == SLEEP_COLUMNS
    pd.testing.assert_frame_equal(sl, rec[SLEEP_COLUMNS])


def test_workouts_schema_and_coherence():
    wk = generate_workouts(n_days=90, seed=5)
    assert list(wk.columns) == WORKOUT_COLUMNS
    assert len(wk) > 100
    # VOLUME = WEIGHT * REPS and REPS_ONLY = REPS
    pd.testing.assert_series_equal(wk["VOLUME"], wk["WEIGHT_LBS"] * wk["REPS"], check_names=False)
    pd.testing.assert_series_equal(wk["REPS_ONLY"], wk["REPS"], check_names=False)
    # RPE only when it was recorded
    assert wk.loc[~wk["RECORDED_RPE"], "RPE"].isna().all()
    assert wk.loc[wk["RECORDED_RPE"], "RPE"].between(6, 10).all()
    assert wk["SET_ORDER"].str.isdigit().all()


# ---------------------------------------------------------------------------
# Capa 2: los CSV commiteados y su contrato con la app
# ---------------------------------------------------------------------------
def test_committed_csvs_exist():
    for name in ["clean_strong_workouts.csv", "clean_sleep_data.csv", "clean_recovery_data.csv"]:
        assert (SYNTHETIC_DIR / name).is_file(), (
            f"missing data/synthetic/{name} (run make synthetic)"
        )


def test_committed_csvs_pass_normalize():
    wk = normalize_workouts(pd.read_csv(SYNTHETIC_DIR / "clean_strong_workouts.csv"))
    sl = normalize_sleep(pd.read_csv(SYNTHETIC_DIR / "clean_sleep_data.csv"))
    rec = normalize_recovery(pd.read_csv(SYNTHETIC_DIR / "clean_recovery_data.csv"))
    assert {"Date", "est_1RM"}.issubset(wk.columns)  # normalize derived columns
    assert len(sl) > 0 and len(rec) > 0


def test_committed_recovery_supports_models_tab():
    # Replica of the prep in app/tabs/models/__init__.py, including the
    # classification guard (needs >= 50 rows: 40 test + 10 train).
    rec = normalize_recovery(pd.read_csv(SYNTHETIC_DIR / "clean_recovery_data.csv"))
    rec["Date"] = pd.to_datetime(rec["Date"], errors="coerce")
    rec["Sleep_need_hrs"] = rec["Sleep Need"].apply(string_to_decimal_hours)
    rec["Efficiency"] = rec["Efficiency"].str.replace("%", "").astype(float)
    rec["Sleep_hr_surplus"] = rec["Asleep hrs"] - rec["Sleep_need_hrs"]
    df_model = rec[["Date"] + MODEL_PREDICTORS + ["Score", "Quality"]].dropna()
    assert len(df_model) >= 50, "the classification guard would cut off the demo"
    y = (df_model["Score"] < 80).astype(int)
    assert y.nunique() == 2, "the binary target needs both classes"
    minority = y.mean() if y.mean() < 0.5 else 1 - y.mean()
    assert 0.05 <= minority <= 0.45, "imbalance outside a realistic range"


def test_committed_workouts_are_hierarchical():
    wk = pd.read_csv(SYNTHETIC_DIR / "clean_strong_workouts.csv")
    sets_per_workout = wk.groupby("DATE").size()
    assert (sets_per_workout >= 6).all(), "each session must have several sets"
    per_workout_names = wk.groupby("DATE")["WORKOUT_NAME"].nunique()
    assert (per_workout_names == 1).all(), "one session = a single WORKOUT_NAME"


def test_no_real_data_dates_leak():
    # The synthetic dataset spans exactly N_DAYS ending on the fixed END_DATE:
    # no row should fall outside the range declared by the generator.
    from rehab_strength.synthetic.generate import END_DATE, N_DAYS

    rec = pd.read_csv(SYNTHETIC_DIR / "clean_recovery_data.csv")
    dates = pd.to_datetime(rec["Date"])
    assert dates.max() == pd.Timestamp(END_DATE)
    assert len(rec) == N_DAYS


@pytest.mark.parametrize("csv", ["clean_sleep_data.csv", "clean_recovery_data.csv"])
def test_committed_source_is_marked_synthetic(csv):
    df = pd.read_csv(SYNTHETIC_DIR / csv)
    assert (df["Data Source"] == "Synthetic").all()
