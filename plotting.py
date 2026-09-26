"""Grafici della campagna.

Si usa ``matplotlib.figure.Figure`` direttamente, senza ``pyplot``: nessun
registro globale di figure, quindi nessun leak quando la dashboard Streamlit
ri-esegue lo script a ogni interazione (D-02), e nessun backend GUI richiesto
nei processi batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from .stats import DispersionEllipse, running_mean_band
from .status import FlightStatus

LABELS = {FlightStatus.OK: "Recupero nominale", FlightStatus.BALLISTIC: "Impatto balistico (main non aperto)"}


@dataclass(frozen=True, slots=True)
class Palette:
    """Colori di una figura. I due colori di serie restano distinguibili anche con daltonismo."""

    surface: str
    ink: str
    muted: str
    grid: str
    ok: str
    ballistic: str
    band: str

    def series(self, status: FlightStatus) -> str:
        return self.ok if status is FlightStatus.OK else self.ballistic


LIGHT = Palette(
    surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e4e3df", ok="#2a78d6", ballistic="#eb6834", band="#cde2fb"
)
"""Tema chiaro: usato per i PNG del report, pensati per la stampa."""

DARK = Palette(
    surface="#0d1220", ink="#e6ecf8", muted="#9aa7bd", grid="#1e2a44", ok="#3987e5", ballistic="#d95926", band="#17324f"
)
"""Tema scuro della dashboard."""


def _style(fig: Figure, palette: Palette) -> None:
    fig.set_facecolor(palette.surface)
    for ax in fig.axes:
        ax.set_facecolor(palette.surface)
        ax.grid(True, color=palette.grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(palette.muted)
        ax.tick_params(colors=palette.muted, labelsize=9)
        ax.xaxis.label.set_color(palette.ink)
        ax.yaxis.label.set_color(palette.ink)
        ax.title.set_color(palette.ink)


def plot_dispersion(
    df: pd.DataFrame,
    ellipse: DispersionEllipse | None,
    nominal_impact: tuple[float, float] | None = None,
    palette: Palette = LIGHT,
) -> Figure:
    """Mappa degli impatti (Est/Nord rispetto alla rampa) con ellisse di predizione."""
    fig = Figure(figsize=(7.5, 7.5), layout="constrained")
    ax = fig.subplots()
    status = df["status"].astype(str)
    for st in (FlightStatus.OK, FlightStatus.BALLISTIC):
        sel = df[status == st.value]
        if len(sel):
            ax.scatter(
                sel["impact_x_m"],
                sel["impact_y_m"],
                s=10,
                alpha=0.55,
                color=palette.series(st),
                edgecolors="none",
                label=f"{LABELS[st]} (n={len(sel)})",
            )
    ax.scatter([0], [0], marker="^", s=110, color=palette.ink, label="Rampa di lancio", zorder=5)
    if nominal_impact is not None:
        ax.scatter(
            *nominal_impact, marker="x", s=80, color=palette.ink, linewidths=2, label="Impatto nominale", zorder=5
        )
    if ellipse is not None:
        ax.add_patch(
            Ellipse(
                ellipse.center,
                2 * ellipse.semi_major_m,
                2 * ellipse.semi_minor_m,
                angle=ellipse.angle_deg,
                fill=False,
                edgecolor=palette.ink,
                linewidth=1.5,
                linestyle="--",
                label=(
                    f"Ellisse di predizione {ellipse.level:.0%} (copertura empirica {ellipse.empirical_coverage:.1%})"
                ),
            )
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Est [m]")
    ax.set_ylabel("Nord [m]")
    ax.set_title("Punti di impatto rispetto alla rampa", loc="left", fontsize=12)
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=palette.ink)
    _style(fig, palette)
    return fig


def plot_convergence(apogee_agl_m: np.ndarray | pd.Series, palette: Palette = LIGHT) -> Figure:
    """Media progressiva dell'apogeo con banda al 95 %: mostra se N è sufficiente."""
    n, mean, half = running_mean_band(np.asarray(apogee_agl_m, dtype=float))
    fig = Figure(figsize=(7.5, 3.6), layout="constrained")
    ax = fig.subplots()
    ax.fill_between(n, mean - half, mean + half, color=palette.band, linewidth=0, label="IC 95 % della media")
    ax.plot(n, mean, color=palette.ok, linewidth=2, label="Media progressiva")
    if n.size > 20:
        lo, hi = np.nanmin((mean - half)[10:]), np.nanmax((mean + half)[10:])
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Numero di voli")
    ax.set_ylabel("Apogeo AGL [m]")
    ax.set_title("Convergenza della stima dell'apogeo medio", loc="left", fontsize=12)
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=palette.ink)
    _style(fig, palette)
    return fig


def plot_histogram(
    values: np.ndarray | pd.Series, label: str, unit: str, bins: int = 30, palette: Palette = LIGHT
) -> Figure:
    """Distribuzione di una grandezza, con media e 95esimo percentile evidenziati."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    fig = Figure(figsize=(7.5, 4.0), layout="constrained")
    ax = fig.subplots()
    if data.size:
        ax.hist(data, bins=bins, color=palette.ok, alpha=0.85, edgecolor=palette.surface, linewidth=0.8)
        mean, p95 = float(np.mean(data)), float(np.quantile(data, 0.95))
        ax.axvline(mean, color=palette.ink, linestyle="--", linewidth=1.5, label=f"Media: {mean:,.1f} {unit}")
        ax.axvline(
            p95,
            color=palette.ballistic,
            linestyle="--",
            linewidth=1.5,
            label=f"95esimo percentile: {p95:,.1f} {unit}",
        )
        ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=palette.ink)
    ax.set_xlabel(f"{label} [{unit}]" if unit else label)
    ax.set_ylabel("Numero di voli")
    ax.set_title(f"Distribuzione: {label.lower()}", loc="left", fontsize=12)
    _style(fig, palette)
    return fig


def save_figure(fig: Figure, path: Path, dpi: int = 200) -> None:
    """Salva su file temporaneo e rinomina: mai un PNG troncato su disco."""
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    tmp.replace(path)
