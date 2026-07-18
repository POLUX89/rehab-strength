"""Tests de los helpers estadísticos extraídos del monolito a app/helpers/stats.py."""

import numpy as np
import pandas as pd
import pytest

from app.helpers import stats


@pytest.fixture
def normal_series():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(50, 10, 300))


def test_normality_test_detects_normal(normal_series):
    p, interpretation = stats.normality_test(normal_series)
    assert 0 <= p <= 1
    assert "normally distributed" in interpretation


def test_normality_test_detects_non_normal():
    skewed = pd.Series(np.concatenate([np.zeros(100), np.arange(1, 100) ** 2]))
    p, interpretation = stats.normality_test(skewed)
    assert p < 0.05
    assert "not normally distributed" in interpretation


def test_iqr_flags_extreme_value():
    s = pd.Series([10, 11, 12, 13, 14, 200])
    outliers = stats.outlier_dectection_iqr(s)
    assert 200 in outliers.values


def test_zscore_modified_flags_extreme_value():
    s = pd.Series([10, 11, 12, 13, 14, 15, 500])
    outliers = stats.outlier_detection_zscore_modified(s)
    assert 500 in outliers.values


def test_compute_ecdf_complementary():
    data = np.arange(1, 101)  # 1..100
    # P(X > 50) complementario ≈ 0.5
    assert stats.compute_ecdf(data, 50, complementary=True) == pytest.approx(0.5, abs=0.01)
    # no complementario: P(X <= 50) ≈ 0.5
    assert stats.compute_ecdf(data, 50, complementary=False) == pytest.approx(0.5, abs=0.01)


def test_fit_distribution_returns_ranked_frame():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 500)
    result = stats.fit_distribution(data)
    assert list(result.columns) == ["distribution", "params", "AIC", "BIC"]
    # ordenado ascendente por AIC
    assert result["AIC"].is_monotonic_increasing


def test_metrics_learning_curve_happy_path():
    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame(
        {
            "Score": rng.normal(70, 5, n),
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        }
    )
    out = stats.metrics_learning_curve(df, n, ["x1", "x2"], H=30, min_train=10)
    assert out["Train size"] == n - 30
    assert out["Test size"] == 30
    assert "Test R²" in out


def test_metrics_learning_curve_skips_when_too_few_rows():
    df = pd.DataFrame({"Score": [1, 2, 3], "x1": [1, 2, 3]})
    assert stats.metrics_learning_curve(df, 3, ["x1"], H=30, min_train=10) is None
