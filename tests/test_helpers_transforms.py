"""Tests de los helpers de transformación extraídos a app/helpers/transforms.py."""

import numpy as np
import pandas as pd
import pytest

from app.helpers import transforms as t


def test_make_unique_columns_suffixes_dupes():
    assert t.make_unique_columns(["a", "a", "b", "a"]) == ["a", "a.1", "b", "a.2"]


def test_make_unique_columns_strips():
    assert t.make_unique_columns([" x ", "y"]) == ["x", "y"]


def test_pick_col_first_match():
    df = pd.DataFrame(columns=["b", "c"])
    assert t.pick_col(df, ["a", "b", "c"]) == "b"
    assert t.pick_col(df, ["z"]) is None
    assert t.pick_col(None, ["a"]) is None


def test_epley_1rm():
    assert t.epley_1rm(100, 5) == pytest.approx(100 * (1 + 5 / 30))
    assert np.isnan(t.epley_1rm(np.nan, 5))
    assert np.isnan(t.epley_1rm("x", 5))


@pytest.mark.parametrize(
    ("x", "prefix"),
    [(0.8, "🟢"), (0.6, "🟡"), (0.3, "🔴")],
)
def test_recovery_zone(x, prefix):
    assert t.recovery_zone(x).startswith(prefix)


def test_recovery_zone_no_data():
    assert t.recovery_zone(None) == "No data"
    assert t.recovery_zone(np.nan) == "No data"


def test_sleep_classifier():
    assert t.sleep_classifier("Good") == 1
    assert t.sleep_classifier("Excellent") == 1
    assert t.sleep_classifier("Poor") == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1h 30min", 1.5), ("2h", 2.0), ("45min", 0.75)],
)
def test_string_to_decimal_hours(raw, expected):
    assert t.string_to_decimal_hours(raw) == pytest.approx(expected)


def test_string_to_decimal_hours_na():
    assert np.isnan(t.string_to_decimal_hours(np.nan))


def test_week_bounds_monday_to_sunday():
    start, end = t.week_bounds("2026-07-15")  # miércoles
    assert start.strftime("%A") == "Monday"
    assert end.strftime("%A") == "Sunday"
    assert (end - start).days == 6


def test_safe_minimal_last_returns_last_valid():
    df = pd.DataFrame(
        {"Date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-02"]), "v": [1, 3, 2]}
    )
    assert t.safe_minimal_last(df, "Date", "v") == 3  # el de fecha más reciente


def test_safe_minimal_last_guards():
    assert t.safe_minimal_last(None, "Date", "v") is None
    assert t.safe_minimal_last(pd.DataFrame({"Date": []}), "Date", "v") is None


def test_normalize_workouts_adds_est_1rm_and_date():
    df = pd.DataFrame(
        {
            "DATE": ["2026-01-01 10:00:00"],
            "WEIGHT_LBS": ["100"],
            "REPS": ["5"],
            "EXERCISE_NAME": [" Sq "],
        }
    )
    out = t.normalize_workouts(df)
    assert "est_1RM" in out.columns
    assert out["EXERCISE_NAME"].iloc[0] == "Sq"
    assert out["est_1RM"].iloc[0] == pytest.approx(100 * (1 + 5 / 30))


def test_normalize_sleep_renames_date_and_coerces():
    df = pd.DataFrame({"DATE": ["2026-01-01"], "Score": ["80"]})
    out = t.normalize_sleep(df)
    assert "Date" in out.columns
    assert out["Score"].iloc[0] == 80
