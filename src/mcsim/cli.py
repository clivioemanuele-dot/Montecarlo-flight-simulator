"""Interfaccia a riga di comando.

Esempi::

    python -m mcsim -n 1000 --seed 42
    python -m mcsim -n 200 --seed 42 --workers 4 --config config/example.toml --out runs
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import SimConfig, load_config
from .model import FlightInputs
from .plotting import plot_convergence, plot_dispersion, save_figure
from .report import write_report
from .runner import default_workers, run_monte_carlo, simulate_one
from .sampling import FlightTask
from .stats import summarize_run
from .status import FlightStatus

log = logging.getLogger("mcsim")


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mcsim", description="Analisi di dispersione Monte Carlo 6-DOF (RocketPy).")
    p.add_argument("-n", "--n-flights", type=int, default=1000, help="numero di voli (default 1000)")
    p.add_argument("--seed", type=int, required=True, help="seme radice: rende il run riproducibile")
    p.add_argument("--workers", type=int, default=default_workers(), help="processi paralleli (default CPU-1)")
    p.add_argument("--config", type=Path, help="file TOML che sovrascrive i default")
    p.add_argument("--out", type=Path, default=Path("runs"), help="cartella base dei run (default ./runs)")
    p.add_argument("--level", type=float, default=0.95, help="livello delle regioni di confidenza")
    p.add_argument("--no-progress", action="store_true", help="disattiva la barra di avanzamento")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if not 0.5 <= args.level < 1.0:
        log.error("--level deve essere in [0.5, 1)")
        return 2
    cfg = load_config(args.config) if args.config else SimConfig()
    cfg.check_data_files()  # curve di spinta e resistenza presenti: fallire qui costa millisecondi

    nominal = simulate_one(FlightTask(-1, FlightInputs.nominal(cfg), noise_seed=args.seed), cfg)
    if nominal.status is not FlightStatus.OK:
        log.error("Il volo nominale non è valido (%s: %s): controlla il modello", nominal.status, nominal.error)
        return 1
    log.info(
        "Nominale: apogeo AGL %.1f m, margine statico %.2f cal, main aperto a %.1f m/s",
        nominal.apogee_agl_m,
        nominal.static_margin_liftoff_cal,
        nominal.main_deploy_speed_ms,
    )

    run = run_monte_carlo(cfg, args.n_flights, args.seed, args.out, args.workers, progress=not args.no_progress)
    summary = summarize_run(run.results, args.level)

    figures: dict[str, str] = {}
    ellipse = summary.landing_nominal.ellipse if summary.landing_nominal else None
    save_figure(
        plot_dispersion(run.results, ellipse, (nominal.impact_x_m, nominal.impact_y_m)),
        run.run_dir / "dispersion.png",
    )
    figures["dispersion"] = "dispersion.png"
    physical = run.results[run.results["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])]
    if len(physical) > 2:
        save_figure(plot_convergence(physical["apogee_agl_m"]), run.run_dir / "convergence.png")
        figures["convergence"] = "convergence.png"
    report = write_report(run.run_dir, summary, run.metadata, figures)

    a = summary.apogee
    log.info("Apogeo AGL: %.1f m (IC95 %.1f–%.1f), P95 %.1f m", a.mean, *a.mean_ci, a.p95)
    if ellipse is not None and not ellipse.gaussian_consistent:
        log.warning(
            "Ellisse non rappresentativa: copertura empirica %.1f%% vs %.0f%% nominale",
            100 * ellipse.empirical_coverage,
            100 * ellipse.level,
        )
    log.info("Risultati in %s (report: %s)", run.run_dir, report.name)
    failed = run.metadata["status_counts"][FlightStatus.ERROR.value]
    return 0 if failed == 0 else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
