"""Coerenza tecnica delle curve Cd(Mach) e dei file su disco."""

import numpy as np

from mcsim.aero import drag_curves, mach_grid, write_curves
from mcsim.config import RocketConfig, SimConfig, airframe_from_config

AIRFRAME = airframe_from_config(RocketConfig())


def test_subsonic_drag_in_realistic_range() -> None:
    """Un razzo sperimentale snello sta tra 0,35 e 0,60 di Cd in regime subsonico."""
    cd_off, _ = drag_curves(AIRFRAME, np.array([0.2, 0.3, 0.5, 0.7]))
    assert np.all((cd_off > 0.35) & (cd_off < 0.60))


def test_transonic_rise() -> None:
    cd_off, _ = drag_curves(AIRFRAME, np.array([0.80, 1.05]))
    assert cd_off[1] / cd_off[0] > 1.3  # la resistenza sale almeno del 30 % attraverso il transonico


def test_power_on_drag_is_lower_everywhere() -> None:
    """A motore acceso manca la resistenza di base: differenza fisica, non un fattore arbitrario."""
    mach = mach_grid()
    cd_off, cd_on = drag_curves(AIRFRAME, mach)
    assert np.all(cd_on < cd_off)
    assert np.all(np.isfinite(cd_off)) and np.all(cd_on > 0)


def test_base_drag_grows_with_mach() -> None:
    mach = np.array([0.2, 0.5, 0.79])
    cd_off, cd_on = drag_curves(AIRFRAME, mach)
    base = cd_off - cd_on
    assert np.all(np.diff(base) > 0)


def test_curves_on_disk_match_the_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """I CSV versionati devono essere rigenerabili: niente numeri orfani nel repository."""
    cfg = SimConfig()
    regenerated = write_curves(AIRFRAME, tmp_path)
    for produced, committed in zip(
        regenerated, (cfg.rocket.power_off_drag_csv, cfg.rocket.power_on_drag_csv), strict=True
    ):
        assert produced.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")


def test_rejects_non_positive_mach() -> None:
    try:
        drag_curves(AIRFRAME, np.array([0.0, 0.5]))
    except ValueError:
        return
    raise AssertionError("atteso ValueError per Mach nullo")
