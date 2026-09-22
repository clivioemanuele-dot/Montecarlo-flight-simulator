from dataclasses import replace

import pytest

from mcsim.config import SimConfig
from mcsim.distributions import Dist
from mcsim.sampling import make_tasks


def test_same_seed_same_tasks() -> None:
    cfg = SimConfig()
    assert make_tasks(cfg, 50, seed=42) == make_tasks(cfg, 50, seed=42)


def test_different_seed_different_tasks() -> None:
    cfg = SimConfig()
    assert make_tasks(cfg, 5, seed=1)[0].inputs != make_tasks(cfg, 5, seed=2)[0].inputs


def test_prefix_stable_when_n_grows() -> None:
    cfg = SimConfig()
    assert make_tasks(cfg, 10, seed=3) == make_tasks(cfg, 100, seed=3)[:10]


def test_adding_a_parameter_does_not_change_the_others() -> None:
    base = SimConfig()
    reduced = replace(base, stochastic={k: v for k, v in base.stochastic.items() if k != "drag_scale"})
    a = make_tasks(base, 5, seed=9)
    b = make_tasks(reduced, 5, seed=9)
    for ta, tb in zip(a, b, strict=True):
        assert ta.inputs.wind_speed_ref_ms == tb.inputs.wind_speed_ref_ms
        assert ta.inputs.inclination_deg == tb.inputs.inclination_deg
        assert tb.inputs.drag_scale == 1.0  # nominale


def test_heading_wrapped() -> None:
    cfg = SimConfig()
    headings = [t.inputs.heading_deg for t in make_tasks(cfg, 500, seed=0)]
    assert all(0.0 <= h < 360.0 for h in headings)
    assert any(h > 300 for h in headings)  # N(0°, 1°) attraversa 0 → valori prossimi a 360


def test_inclination_never_reaches_vertical() -> None:
    cfg = SimConfig()
    assert max(t.inputs.inclination_deg for t in make_tasks(cfg, 2000, seed=0)) < 90.0


def test_unknown_stochastic_parameter_rejected() -> None:
    cfg = replace(SimConfig(), stochastic={"rail_lenght_m": Dist.normal(5.2, 0.1)})
    with pytest.raises(KeyError):
        make_tasks(cfg, 1, seed=0)


def test_invalid_n_rejected() -> None:
    with pytest.raises(ValueError):
        make_tasks(SimConfig(), 0, seed=0)
