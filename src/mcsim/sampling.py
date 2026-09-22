"""Campionamento riproducibile degli input stocastici.

Schema dei semi (P1-03):

- ogni coppia (volo ``i``, parametro ``p``) ha un proprio flusso
  ``SeedSequence(seed, spawn_key=(i, crc32(p)))``;
- quindi il campione del volo ``i`` dipende solo da ``(seed, i, p)``: non cambia
  con il numero di worker, con l'ordine di esecuzione, con ``N`` né quando si
  aggiunge o rimuove un altro parametro incerto (utile per confronti a numeri
  casuali comuni tra due configurazioni);
- un ulteriore flusso per volo alimenta ``np.random.seed`` nel worker, perché
  RocketPy 1.x genera il rumore del sensore del paracadute con l'RNG globale.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, fields, replace

import numpy as np

from .config import SimConfig
from .model import FlightInputs

_NOISE_STREAM = "rocketpy_global_noise"
INPUT_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FlightInputs))


@dataclass(frozen=True, slots=True)
class FlightTask:
    """Unità di lavoro serializzabile inviata ai worker."""

    index: int
    inputs: FlightInputs
    noise_seed: int


def _stream(seed: int, index: int, name: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed, spawn_key=(index, zlib.crc32(name.encode("utf-8"))))


def validate_stochastic_model(cfg: SimConfig) -> None:
    """Rifiuta parametri incerti che non esistono in ``FlightInputs`` (refusi silenziosi)."""
    unknown = sorted(set(cfg.stochastic) - set(INPUT_FIELDS))
    if unknown:
        raise KeyError(f"Parametri stocastici sconosciuti: {unknown}. Validi: {list(INPUT_FIELDS)}")
    if cfg.rocket.drogue is None:
        dead = sorted({"drogue_cds_scale", "drogue_lag_s"} & set(cfg.stochastic))
        if dead:
            raise KeyError(f"{dead} definiti ma il razzo non ha drogue")


def sample_inputs(cfg: SimConfig, seed: int, index: int) -> FlightInputs:
    """Input del volo ``index``: valori nominali sovrascritti dai parametri incerti."""
    values = {
        name: dist.sample(np.random.default_rng(_stream(seed, index, name))) for name, dist in cfg.stochastic.items()
    }
    if "heading_deg" in values:
        values["heading_deg"] %= 360.0
    return replace(FlightInputs.nominal(cfg), **values)


def make_tasks(cfg: SimConfig, n: int, seed: int) -> list[FlightTask]:
    """Genera ``n`` task deterministici a partire da ``seed``."""
    if n <= 0:
        raise ValueError(f"Il numero di voli deve essere positivo, ricevuto {n}")
    if seed < 0:
        raise ValueError("Il seed deve essere un intero non negativo")
    validate_stochastic_model(cfg)
    return [
        FlightTask(
            index=i,
            inputs=sample_inputs(cfg, seed, i),
            noise_seed=int(_stream(seed, i, _NOISE_STREAM).generate_state(1)[0]),
        )
        for i in range(n)
    ]
