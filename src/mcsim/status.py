"""Esiti possibili di un volo (modulo senza dipendenze da RocketPy)."""

from __future__ import annotations

from enum import StrEnum


class FlightStatus(StrEnum):
    OK = "ok"
    """Volo completo con apertura del main."""
    BALLISTIC = "ballistic"
    """Atterrato senza apertura del main: fisicamente valido, critico per la safety."""
    NO_RAIL_EXIT = "no_rail_exit"
    """Il razzo non lascia la rampa (spinta/peso insufficiente)."""
    NOT_LANDED = "not_landed"
    """Integrazione terminata per ``max_time`` prima dell'impatto."""
    NON_FINITE = "non_finite"
    """NaN o infinito nelle grandezze estratte: divergenza numerica."""
    ERROR = "error"
    """Eccezione durante costruzione del modello o integrazione."""


PHYSICAL_STATUSES: frozenset[FlightStatus] = frozenset({FlightStatus.OK, FlightStatus.BALLISTIC})
"""Esiti che rappresentano un volo reale e alimentano le statistiche di dispersione."""
