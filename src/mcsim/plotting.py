"""Grafici della campagna.

Si usa ``matplotlib.figure.Figure`` direttamente, senza ``pyplot``: nessun
registro globale di figure, quindi nessun leak quando la dashboard Streamlit
ri-esegue lo script a ogni interazione (D-02), e nessun backend GUI richiesto
nei processi batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from .stats import DispersionEllipse, running_mean_band
from .status import FlightStatus

# Palette di riferimento (slot categorici 1-2, validati all-pairs per scatter) + inchiostri neutri.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {FlightStatus.OK: "#2a78d6", FlightStatus.BALLISTIC: "#eb6834"}
BAND = "#cde2fb"
LABELS = {FlightStatus.OK: "Recupero nominale", FlightStatus.BALLISTIC: "Impatto balistico (main non aperto)"}


def _style(fig: Figure) -> None:
    fig.set_facecolor(SURFACE)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)


def plot_dispersion(
    df: pd.DataFrame,
    ellipse: DispersionEllipse | None,
    nominal_impact: tuple[float, float] | None = None,
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
                color=SERIES[st],
                edgecolors="none",
                label=f"{LABELS[st]} (n={len(sel)})",
            )
    ax.scatter([0], [0], marker="^", s=90, color=INK, label="Rampa", zorder=5)
    if nominal_impact is not None:
        ax.scatter(*nominal_impact, marker="x", s=80, color=INK, linewidths=2, label="Impatto nominale", zorder=5)
    if ellipse is not None:
        ax.add_patch(
            Ellipse(
                ellipse.center,
                2 * ellipse.semi_major_m,
                2 * ellipse.semi_minor_m,
                angle=ellipse.angle_deg,
                fill=False,
                edgecolor=INK,
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
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK)
    _style(fig)
    return fig


def plot_convergence(apogee_agl_m: np.ndarray | pd.Series) -> Figure:
    """Media progressiva dell'apogeo con banda al 95 %: mostra se N è sufficiente."""
    n, mean, half = running_mean_band(np.asarray(apogee_agl_m, dtype=float))
    fig = Figure(figsize=(7.5, 3.6), layout="constrained")
    ax = fig.subplots()
    ax.fill_between(n, mean - half, mean + half, color=BAND, linewidth=0, label="IC 95 % della media")
    ax.plot(n, mean, color=SERIES[FlightStatus.OK], linewidth=2, label="Media progressiva")
    if n.size > 20:
        lo, hi = np.nanmin((mean - half)[10:]), np.nanmax((mean + half)[10:])
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Numero di voli")
    ax.set_ylabel("Apogeo AGL [m]")
    ax.set_title("Convergenza della stima dell'apogeo medio", loc="left", fontsize=12)
    ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK)
    _style(fig)
    return fig


def save_figure(fig: Figure, path: Path, dpi: int = 200) -> None:
    """Salva su file temporaneo e rinomina: mai un PNG troncato su disco."""
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
    tmp.replace(path)
