import math

import numpy as np
import pandas as pd
import pytest

from mcsim.stats import dispersion_ellipse, ellipse_scale, proportion_ci, quantile_ci, summarize_run


def test_chi2_scale_95() -> None:
    assert math.isclose(ellipse_scale(10**6, 0.95, "chi2"), 2.4477, abs_tol=1e-4)


def test_prediction_scale_larger_for_small_n() -> None:
    assert ellipse_scale(10, 0.95) > 3.3
    assert math.isclose(ellipse_scale(1000, 0.95), ellipse_scale(1000, 0.95, "chi2"), rel_tol=5e-3)


def test_gaussian_coverage_matches_level() -> None:
    rng = np.random.default_rng(0)
    pts = rng.multivariate_normal([100.0, -50.0], [[400.0, 150.0], [150.0, 100.0]], size=5000)
    e = dispersion_ellipse(pts[:, 0], pts[:, 1], 0.95)
    assert e.gaussian_consistent
    assert abs(e.empirical_coverage - 0.95) < 0.01


def test_ring_distribution_flagged_as_non_gaussian() -> None:
    """Riproduce il difetto P1-02: impatti ad anello (heading uniforme)."""
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 1000)
    r = rng.normal(700, 80, 1000)
    e = dispersion_ellipse(r * np.cos(theta), r * np.sin(theta), 0.95)
    assert not e.gaussian_consistent


def test_one_sigma_ellipse_covers_39_percent_in_2d() -> None:
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((200_000, 2))
    assert math.isclose(np.mean(np.sum(pts**2, axis=1) <= 1.0), 0.3935, abs_tol=0.005)


def test_quantile_ci_contains_true_quantile() -> None:
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(400):
        lo, hi = quantile_ci(rng.standard_normal(300), 0.95)
        hits += lo <= 1.6449 <= hi
    assert hits / 400 > 0.92


def test_quantile_ci_infinite_when_n_too_small() -> None:
    assert math.isinf(quantile_ci(np.arange(10.0), 0.95)[1])


def test_proportion_ci_edges() -> None:
    lo, hi = proportion_ci(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04


def test_summarize_run_excludes_failures() -> None:
    df = pd.DataFrame(
        {
            "status": ["ok"] * 5 + ["no_rail_exit"],
            "apogee_agl_m": [2000.0, 2010.0, 1990.0, 2005.0, 1995.0, 0.0],
            "max_mach": [0.6] * 5 + [np.nan],
            "main_deploy_speed_ms": [16.0] * 5 + [np.nan],
            "static_margin_liftoff_cal": [2.3] * 5 + [np.nan],
            "impact_x_m": [10.0, 20.0, 15.0, 12.0, 30.0, 0.0],
            "impact_y_m": [5.0, 25.0, 8.0, 30.0, 12.0, 0.0],
        }
    )
    s = summarize_run(df)
    assert s.apogee.n == 5 and s.apogee.minimum == 1990.0  # l'apogeo 0 del razzo fermo non entra
    assert s.status_counts["no_rail_exit"] == 1


def test_degenerate_covariance_raises() -> None:
    with pytest.raises(ValueError):
        dispersion_ellipse(np.ones(10), np.ones(10))
