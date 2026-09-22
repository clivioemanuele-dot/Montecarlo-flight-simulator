"""Esecuzione della campagna Monte Carlo: parallelismo, isolamento errori, classificazione.

Differenze chiave rispetto al loop originale:

- **Isolamento (P2-01):** ogni volo è racchiuso in ``try/except``; un'eccezione
  diventa un record ``status="error"`` e il batch prosegue.
- **Esiti silenziosi (P1-06):** RocketPy non solleva eccezioni se il razzo non
  lascia la rampa o non atterra entro ``max_time``: restituisce apogeo 0 e
  impatto (0, 0). Questi casi sono classificati ed esclusi dalle statistiche.
- **Voli anomali ma fisici** (main non aperto → impatto balistico) restano nel
  dataset con il loro status: scartarli introdurrebbe un bias di sopravvivenza
  proprio sul caso più critico per la sicurezza.
- **Parallelismo (P3-01):** ``ProcessPoolExecutor`` con contesto ``spawn``
  (lo stesso di Windows su tutte le piattaforme); i risultati non dipendono dal
  numero di worker grazie ai semi per-volo.
- **Checkpoint (P2-01):** ogni risultato è scritto su disco appena arriva.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import __version__, _compat
from .config import SimConfig
from .model import FlightInputs, build_flight
from .sampling import FlightTask, make_tasks
from .status import FlightStatus
from .storage import (
    METADATA_FILE,
    PARTIAL_FILE,
    RESULTS_FILE,
    CheckpointWriter,
    atomic_write_csv,
    atomic_write_json,
    prepare_run_dir,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FlightResult:
    """Esito di un volo. Le grandezze non disponibili restano ``NaN``."""

    index: int
    status: FlightStatus
    inputs: FlightInputs
    apogee_agl_m: float = math.nan
    apogee_time_s: float = math.nan
    max_speed_ms: float = math.nan
    max_mach: float = math.nan
    rail_exit_speed_ms: float = math.nan
    static_margin_liftoff_cal: float = math.nan
    main_deploy_speed_ms: float = math.nan
    impact_x_m: float = math.nan
    impact_y_m: float = math.nan
    impact_speed_ms: float = math.nan
    flight_time_s: float = math.nan
    wall_time_s: float = math.nan
    error: str = ""

    def to_row(self) -> dict[str, Any]:
        """Record piatto per CSV: gli input hanno prefisso ``in_``."""
        row: dict[str, Any] = {"index": self.index, "status": self.status.value}
        row.update({f"in_{k}": v for k, v in asdict(self.inputs).items()})
        for name in (
            "apogee_agl_m",
            "apogee_time_s",
            "max_speed_ms",
            "max_mach",
            "rail_exit_speed_ms",
            "static_margin_liftoff_cal",
            "main_deploy_speed_ms",
            "impact_x_m",
            "impact_y_m",
            "impact_speed_ms",
            "flight_time_s",
            "wall_time_s",
        ):
            row[name] = getattr(self, name)
        row["error"] = self.error
        return row


@dataclass(frozen=True, slots=True)
class RunOutput:
    run_dir: Path
    results: pd.DataFrame
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------- singolo volo
def _classify(task: FlightTask, flight: Any, cfg: SimConfig) -> FlightResult:
    """Estrae le metriche e classifica l'esito (nessuna eccezione attesa qui)."""
    if flight.out_of_rail_time <= 0.0:
        return FlightResult(
            task.index, FlightStatus.NO_RAIL_EXIT, task.inputs, error="il razzo non ha lasciato la rampa"
        )
    if flight.t_final >= cfg.launch.max_time_s - 1e-6:
        return FlightResult(
            task.index, FlightStatus.NOT_LANDED, task.inputs, error=f"nessun impatto entro {cfg.launch.max_time_s} s"
        )

    main_events = [t for t, chute in flight.parachute_events if chute.name == cfg.rocket.main.name]
    result = FlightResult(
        index=task.index,
        status=FlightStatus.OK if main_events else FlightStatus.BALLISTIC,
        inputs=task.inputs,
        apogee_agl_m=float(flight.apogee) - cfg.site.elevation_m,  # FIX P1-13: AGL, non ASL
        apogee_time_s=float(flight.apogee_time),
        max_speed_ms=float(flight.max_speed),
        max_mach=float(flight.max_mach_number),
        rail_exit_speed_ms=float(flight.out_of_rail_velocity),
        static_margin_liftoff_cal=float(flight.rocket.static_margin(0.0)),
        main_deploy_speed_ms=float(flight.speed(main_events[0])) if main_events else math.nan,
        impact_x_m=float(flight.x_impact),
        impact_y_m=float(flight.y_impact),
        impact_speed_ms=abs(float(flight.impact_velocity)),
        flight_time_s=float(flight.t_final),
    )
    required = {name: getattr(result, name) for name in _REQUIRED_FINITE}
    bad = sorted(name for name, value in required.items() if not math.isfinite(value))
    if bad:
        return replace(result, status=FlightStatus.NON_FINITE, error=f"valori non finiti: {bad}")
    return result


_REQUIRED_FINITE: tuple[str, ...] = (
    "apogee_agl_m",
    "max_speed_ms",
    "static_margin_liftoff_cal",
    "impact_x_m",
    "impact_y_m",
    "impact_speed_ms",
    "flight_time_s",
)


def simulate_one(task: FlightTask, cfg: SimConfig) -> FlightResult:
    """Simula un volo in isolamento. Non solleva mai eccezioni (tranne BaseException)."""
    # RocketPy 1.x usa l'RNG globale legacy per il rumore del sensore del paracadute:
    # senza questo seed, input identici producono traiettorie diverse (P1-03).
    np.random.seed(task.noise_seed)  # noqa: NPY002 - unico modo di controllare il rumore interno di RocketPy
    t0 = time.perf_counter()
    try:
        flight = build_flight(cfg, task.inputs)
        result = _classify(task, flight, cfg)
    except Exception as exc:  # noqa: BLE001 - isolamento voluto: un volo non abbatte il batch
        return FlightResult(
            task.index,
            FlightStatus.ERROR,
            task.inputs,
            wall_time_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}"[:500],
        )
    return replace(result, wall_time_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------- campagna
def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "mcsim": __version__}
    for pkg in ("rocketpy", "numpy", "scipy", "matplotlib", "pandas"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            out[pkg] = "n/d"
    return out


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 1)


def run_monte_carlo(
    cfg: SimConfig,
    n: int,
    seed: int,
    out_dir: Path | str = "runs",
    workers: int | None = None,
    progress: bool = True,
) -> RunOutput:
    """Esegue ``n`` voli e salva ``results.csv`` e ``metadata.json`` in una nuova cartella di run.

    Parameters
    ----------
    cfg
        Configurazione (modello nominale + modello stocastico).
    n
        Numero di voli.
    seed
        Seme radice: stesso ``(cfg, n, seed)`` ⇒ stesso ``results.csv``, con qualunque ``workers``.
    out_dir
        Cartella base dei run.
    workers
        Processi paralleli; ``1`` esegue in serie nel processo corrente. Default: CPU - 1.
    progress
        Mostra la barra di avanzamento.
    """
    workers = default_workers() if workers is None else workers
    if workers < 1:
        raise ValueError("workers deve essere >= 1")
    tasks = make_tasks(cfg, n, seed)  # valida n, seed e modello stocastico prima di toccare il disco
    run_dir = prepare_run_dir(out_dir, seed, n)
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    worker = partial(simulate_one, cfg=cfg)
    results: list[FlightResult] = []
    log.info("Run %s: %d voli, seed=%d, workers=%d", run_dir.name, n, seed, workers)

    with (
        CheckpointWriter(run_dir / PARTIAL_FILE) as ckpt,
        tqdm(total=n, disable=not progress, desc="Voli", unit="volo") as bar,
    ):

        def collect(result: FlightResult) -> None:
            ckpt.write(result.to_row())
            results.append(result)
            bar.update()
            if result.status is FlightStatus.ERROR:
                log.warning("Volo %d fallito: %s", result.index, result.error)

        if workers == 1:
            for task in tasks:
                collect(worker(task))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
                futures = {pool.submit(worker, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        collect(future.result())
                    except Exception as exc:  # noqa: BLE001 - worker morto (segfault, OOM), pickling
                        collect(
                            FlightResult(task.index, FlightStatus.ERROR, task.inputs, error=f"worker: {exc!r}"[:500])
                        )

    results.sort(key=lambda r: r.index)
    df = pd.DataFrame([r.to_row() for r in results])
    atomic_write_csv(df, run_dir / RESULTS_FILE)
    (run_dir / PARTIAL_FILE).unlink(missing_ok=True)

    counts = df["status"].value_counts()
    metadata: dict[str, Any] = {
        "seed": seed,
        "n_flights": n,
        "workers": workers,
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "status_counts": {s.value: int(counts.get(s.value, 0)) for s in FlightStatus},
        "numpy_trapz_shim": _compat.SHIM_APPLIED,
        "versions": _versions(),
        "config": cfg.to_dict(),
    }
    atomic_write_json(run_dir / METADATA_FILE, metadata)
    log.info("Completato in %.1f s: %s", metadata["wall_time_s"], metadata["status_counts"])
    return RunOutput(run_dir, df, metadata)
