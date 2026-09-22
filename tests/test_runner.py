"""Test d'integrazione con RocketPy reale (lenti: ~3 s per volo)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mcsim import runner
from mcsim.config import SimConfig
from mcsim.distributions import Dist
from mcsim.model import FlightInputs
from mcsim.runner import run_monte_carlo, simulate_one
from mcsim.sampling import FlightTask
from mcsim.status import FlightStatus

pytestmark = pytest.mark.slow

NUMERIC = ["apogee_agl_m", "impact_x_m", "impact_y_m", "flight_time_s", "main_deploy_speed_ms"]


def test_nominal_flight_is_physically_sane() -> None:
    """Volo di riferimento: Calisto con Cesaroni M1670, vento nullo, rampa a 85°."""
    cfg = SimConfig()
    r = simulate_one(FlightTask(0, FlightInputs.nominal(cfg), noise_seed=1), cfg)
    assert r.status is FlightStatus.OK
    assert 1.5 < r.static_margin_liftoff_cal < 3.5  # margine di progetto tipico, non i 6 cal senza ogiva
    assert 2500.0 < r.apogee_agl_m < 3600.0  # classe 3 km, coerente con 6026 N·s su ~19,7 kg
    assert 0.6 < r.max_mach < 0.95  # subsonico alto: la curva Cd transonica non viene mai usata a fondo
    assert r.rail_exit_speed_ms > 20.0  # criterio di sicurezza per l'uscita dalla rampa
    assert r.main_deploy_speed_ms < 30.0  # con il drogue: non più ~148 m/s
    assert 3.0 < r.impact_speed_ms < 9.0  # discesa sotto il main
    assert abs(r.impact_x_m) < 10.0  # heading 0°, vento nullo: solo effetti minori (Coriolis)


def test_identical_inputs_identical_outputs() -> None:
    """Nel codice originale lo stesso input dava t_final 153,717 s e poi 154,004 s."""
    cfg = SimConfig()
    task = FlightTask(0, FlightInputs.nominal(cfg), noise_seed=123)
    a, b = simulate_one(task, cfg), simulate_one(task, cfg)
    assert a.flight_time_s == b.flight_time_s and a.impact_y_m == b.impact_y_m


def test_serial_and_parallel_runs_are_identical(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = SimConfig()
    serial = run_monte_carlo(cfg, 4, seed=7, out_dir=tmp_path, workers=1, progress=False)
    parallel = run_monte_carlo(cfg, 4, seed=7, out_dir=tmp_path, workers=2, progress=False)
    pd.testing.assert_frame_equal(serial.results[NUMERIC], parallel.results[NUMERIC])
    assert (serial.run_dir / "results.csv").is_file()
    assert not (serial.run_dir / "results.partial.csv").exists()


def test_wind_pushes_the_landing_downwind() -> None:
    """Vento da ovest a 8 m/s: l'impatto si sposta verso est rispetto al nominale."""
    cfg = SimConfig()
    nominal = simulate_one(FlightTask(0, FlightInputs.nominal(cfg), noise_seed=1), cfg)
    windy_inputs = replace(FlightInputs.nominal(cfg), wind_speed_ref_ms=8.0, wind_from_deg=270.0)
    windy = simulate_one(FlightTask(1, windy_inputs, noise_seed=1), cfg)
    assert windy.status is FlightStatus.OK
    assert windy.impact_x_m > nominal.impact_x_m + 500.0
    assert windy.apogee_agl_m < nominal.apogee_agl_m  # weathercocking: parte della quota si perde


def test_rocket_stuck_on_rail_is_classified_not_counted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nel codice originale finiva nel dataset come apogeo 0 e impatto in (0, 0)."""
    cfg = replace(SimConfig(), stochastic={"impulse_scale": Dist.fixed(0.05)})
    out = run_monte_carlo(cfg, 2, seed=0, out_dir=tmp_path, workers=1, progress=False)
    assert set(out.results["status"]) == {FlightStatus.NO_RAIL_EXIT.value}
    assert out.results["apogee_agl_m"].isna().all()


def test_exception_in_one_flight_does_not_kill_batch(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = runner.build_flight

    def flaky(cfg, inputs):  # type: ignore[no-untyped-def]
        if inputs.heading_deg == cfg_bad_heading:
            raise FloatingPointError("integrazione divergente (simulata)")
        return real(cfg, inputs)

    cfg = replace(SimConfig(), stochastic={"heading_deg": Dist.uniform(0.0, 1.0)})
    from mcsim.sampling import make_tasks

    cfg_bad_heading = make_tasks(cfg, 3, seed=5)[1].inputs.heading_deg
    monkeypatch.setattr(runner, "build_flight", flaky)
    out = run_monte_carlo(cfg, 3, seed=5, out_dir=tmp_path, workers=1, progress=False)
    status = out.results.set_index("index")["status"]
    assert status[1] == FlightStatus.ERROR.value
    assert status[0] == status[2] == FlightStatus.OK.value
    assert "FloatingPointError" in out.results.loc[1, "error"]
    assert np.isfinite(out.results.loc[0, "apogee_agl_m"])
