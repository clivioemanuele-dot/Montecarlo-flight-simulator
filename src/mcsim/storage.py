"""I/O dei run: cartelle univoche, scritture atomiche, checkpoint incrementale (P2-03).

Struttura di un run::

    runs/20261025-120000_seed42_n1000/
        results.csv          # un record per volo, inclusi i falliti (status, error)
        results.partial.csv  # checkpoint durante l'esecuzione, rimosso a fine run
        metadata.json        # seed, N, versioni, configurazione, tempi, esiti
        dispersion.png, convergence.png, report.tex
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pandas as pd

RESULTS_FILE = "results.csv"
PARTIAL_FILE = "results.partial.csv"
METADATA_FILE = "metadata.json"


def prepare_run_dir(base: Path | str, seed: int, n: int) -> Path:
    """Crea una cartella di run univoca e verifica **subito** i permessi di scrittura.

    Fallire qui costa millisecondi; scoprire un permesso mancante dopo 15 minuti
    di simulazione costa l'intero batch.
    """
    base = Path(base).expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}_seed{seed}_n{n}"
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}_seed{seed}_n{n}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    probe = run_dir / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"Cartella di output non scrivibile: {run_dir}") from exc
    return run_dir


def atomic_write_text(path: Path, text: str) -> None:
    """Scrive su file temporaneo nella stessa cartella e poi rinomina (os.replace è atomico)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, df.to_csv(index=False, lineterminator="\n"))


def _json_safe(obj: Any) -> Any:
    """NaN/inf non sono JSON valido: diventano ``null`` invece di rompere il parser a valle."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(_json_safe(data), indent=2, ensure_ascii=False, allow_nan=False, default=str)
    atomic_write_text(path, text + "\n")


class CheckpointWriter:
    """Appende una riga CSV per volo appena completato, con flush immediato.

    Se il processo viene interrotto (crash, Ctrl+C, OOM), i voli già calcolati
    restano su disco invece di andare persi con la lista in memoria.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = path.open("w", encoding="utf-8", newline="")
        self._writer: csv.DictWriter[str] | None = None

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def list_runs(base: Path | str) -> list[Path]:
    """Run completi (con ``results.csv`` e ``metadata.json``), dal più recente."""
    base = Path(base)
    if not base.is_dir():
        return []
    runs = [p for p in base.iterdir() if (p / RESULTS_FILE).is_file() and (p / METADATA_FILE).is_file()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def load_run(run_dir: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / RESULTS_FILE)
    metadata = json.loads((run_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return df, metadata
