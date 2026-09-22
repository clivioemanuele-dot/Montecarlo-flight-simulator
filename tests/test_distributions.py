import math

import numpy as np
import pytest

from mcsim.distributions import Dist


def _draw(dist: Dist, n: int = 20_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([dist.sample(rng) for _ in range(n)])


def test_truncnorm_respects_bounds() -> None:
    x = _draw(Dist.truncnorm(mean=85.0, std=1.5, low=80.0, high=89.5))
    assert x.min() >= 80.0 and x.max() <= 89.5


def test_lognormal_is_strictly_positive_with_given_median() -> None:
    x = _draw(Dist.lognormal(median=1.5, sigma=0.5))
    assert x.min() > 0
    assert math.isclose(np.median(x), 1.5, rel_tol=0.03)


def test_vonmises_wraps_to_circle() -> None:
    x = _draw(Dist.vonmises_deg(mean=355.0, kappa=4.0))
    assert x.min() >= 0.0 and x.max() < 360.0
    mean_dir = math.degrees(math.atan2(np.sin(np.radians(x)).mean(), np.cos(np.radians(x)).mean())) % 360
    assert min(abs(mean_dir - 355.0), 360 - abs(mean_dir - 355.0)) < 2.0


def test_weibull_non_negative() -> None:
    assert _draw(Dist.weibull(shape=2.0, scale=4.0)).min() >= 0.0


def test_same_generator_state_same_sample() -> None:
    d = Dist.truncnorm_sigma(1.0, 0.02)
    assert d.sample(np.random.default_rng(7)) == d.sample(np.random.default_rng(7))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "normal", "mean": 0.0},  # std mancante
        {"kind": "normal", "mean": 0.0, "std": -1.0},
        {"kind": "truncnorm", "mean": 0.0, "std": 1.0, "low": 1.0, "high": 0.0},
        {"kind": "lognormal", "median": -1.0, "sigma": 0.1},
        {"kind": "boh"},
    ],
)
def test_invalid_specs_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Dist(**kwargs)


def test_roundtrip_dict() -> None:
    d = Dist.truncnorm(85.0, 0.5, 83.0, 87.0)
    assert Dist.from_dict(d.to_dict()) == d
