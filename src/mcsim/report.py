"""Report LaTeX della campagna (P2-02).

Garanzie:

- ogni testo proveniente da dati o configurazione passa da :func:`latex_escape`
  (``_``, ``%``, ``&``, ``#``… in nomi di file, errori o parametri romperebbero
  la compilazione);
- ogni numero passa da :func:`fmt`: NaN/inf diventano ``--`` invece di ``nan``,
  il segno meno è in modalità matematica, le migliaia hanno uno spazio sottile;
- scrittura atomica e dipendenze LaTeX minime (``booktabs``, ``graphicx``, ``geometry``, ``float``),
  presenti anche nelle installazioni TeX di base.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .stats import RunSummary, ScalarSummary
from .storage import atomic_write_text

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
_UNICODE_MATH = {"μ": r"$\mu$", "σ": r"$\sigma$", "κ": r"$\kappa$", "λ": r"$\lambda$", "°": r"$^\circ$"}


def latex_escape(text: object) -> str:
    """Rende sicuro per LaTeX un testo arbitrario."""
    out = []
    for ch in str(text):
        out.append(_LATEX_SPECIAL.get(ch) or _UNICODE_MATH.get(ch) or ch)
    return "".join(out)


def fmt(x: float | None, digits: int = 1, pct: bool = False) -> str:
    """Numero in formato LaTeX; ``--`` se mancante o non finito."""
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "--"
    value = 100.0 * x if pct else float(x)
    body = f"{value:,.{digits}f}".replace(",", r"\,")
    suffix = r"\,\%" if pct else ""
    return f"${body}{suffix}$"


def fmt_ci(ci: tuple[float, float], digits: int = 1, pct: bool = False) -> str:
    """Intervallo; un estremo infinito indica N insufficiente per quel livello."""

    def edge(x: float) -> str:
        if isinstance(x, float) and math.isinf(x):
            return r"$+\infty$" if x > 0 else r"$-\infty$"
        return fmt(x, digits, pct)

    return f"[{edge(ci[0])}, {edge(ci[1])}]"


def _scalar_rows(label: str, unit: str, s: ScalarSummary, digits: int = 1) -> list[str]:
    u = latex_escape(unit)
    return [
        rf"\multicolumn{{3}}{{l}}{{\textbf{{{latex_escape(label)}}} (n = {s.n})}} \\",
        rf"Media [{u}] & {fmt(s.mean, digits)} & IC 95\,\%: {fmt_ci(s.mean_ci, digits)} \\",
        rf"Deviazione standard [{u}] & {fmt(s.std, digits)} & \\",
        rf"P5 / P50 / P95 [{u}] & {fmt(s.p05, digits)} / {fmt(s.p50, digits)} / {fmt(s.p95, digits)}"
        rf" & IC 95\,\% P95: {fmt_ci(s.p95_ci, digits)} \\",
        rf"Min / Max [{u}] & {fmt(s.minimum, digits)} / {fmt(s.maximum, digits)} & \\",
    ]


def render_report(summary: RunSummary, metadata: dict[str, Any], figures: dict[str, str]) -> str:
    """Documento LaTeX completo. ``figures`` mappa ``{"dispersion": "dispersion.png", ...}``."""
    versions = metadata.get("versions", {})
    lines: list[str] = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\title{Analisi di dispersione Monte Carlo 6-DOF}",
        rf"\date{{{latex_escape(metadata.get('started_utc', ''))} UTC}}",
        r"\author{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Configurazione del run}",
        r"\begin{table}[H]\centering\begin{tabular}{ll}\toprule",
        rf"Voli simulati & {int(metadata.get('n_flights', summary.n_total))} \\",
        rf"Seed & \texttt{{{latex_escape(metadata.get('seed', '--'))}}} \\",
        rf"Processi paralleli & {latex_escape(metadata.get('workers', '--'))} \\",
        rf"Durata [s] & {fmt(metadata.get('wall_time_s'), 1)} \\",
        rf"RocketPy / NumPy / SciPy & {latex_escape(versions.get('rocketpy', '--'))} / "
        rf"{latex_escape(versions.get('numpy', '--'))} / {latex_escape(versions.get('scipy', '--'))} \\",
        rf"Python & {latex_escape(versions.get('python', '--'))} \\",
        r"\bottomrule\end{tabular}\end{table}",
        r"\section{Esiti dei voli}",
        r"\begin{table}[H]\centering\begin{tabular}{lrr}\toprule",
        r"Esito & Voli & Quota \\ \midrule",
    ]
    for status, count in summary.status_counts.items():
        share = count / summary.n_total if summary.n_total else math.nan
        lines.append(rf"\texttt{{{latex_escape(status)}}} & {count} & {fmt(share, 1, pct=True)} \\")
    lines += [
        r"\bottomrule\end{tabular}\end{table}",
        rf"Tasso di successo: {fmt(summary.success_rate, 1, pct=True)}, IC 95\,\% Clopper--Pearson "
        rf"{fmt_ci(summary.success_rate_ci, 1, pct=True)}. "
        rf"Probabilità di impatto balistico tra i voli fisici: {fmt(summary.ballistic_rate, 2, pct=True)}, "
        rf"IC 95\,\% {fmt_ci(summary.ballistic_rate_ci, 2, pct=True)}. "
        r"I voli con esito numerico non valido sono esclusi dalle statistiche seguenti.",
        r"\section{Prestazioni}",
        r"\begin{table}[H]\centering\begin{tabular}{lll}\toprule",
        *_scalar_rows("Apogeo AGL", "m", summary.apogee),
        r"\midrule",
        *_scalar_rows("Numero di Mach massimo", "-", summary.max_mach, 3),
        r"\midrule",
        *_scalar_rows("Margine statico al decollo", "cal", summary.static_margin, 2),
        r"\midrule",
        *_scalar_rows("Velocità all'apertura del main", "m/s", summary.main_deploy_speed),
        r"\bottomrule\end{tabular}\end{table}",
        r"\section{Dispersione dei punti di impatto}",
    ]
    land = summary.landing_nominal
    if land is None:
        lines.append(r"Voli con recupero nominale insufficienti per stimare la dispersione.")
    else:
        e = land.ellipse
        lines += [
            r"\begin{table}[H]\centering\begin{tabular}{ll}\toprule",
            rf"Voli con recupero nominale & {land.n} \\",
            rf"CEP50 attorno al punto medio [m] & {fmt(land.cep50_m)} \\",
            rf"R95 attorno al punto medio [m] & {fmt(land.r95_m)} \\",
            rf"Distanza dalla rampa, P95 [m] & {fmt(land.range_from_pad.p95)} "
            rf"(IC 95\,\% {fmt_ci(land.range_from_pad.p95_ci)}) \\",
            rf"Distanza massima dalla rampa, tutti i voli fisici [m] & {fmt(summary.max_range_all_physical_m)} \\",
        ]
        if e is not None:
            lines += [
                r"\midrule",
                rf"Centro ellisse (Est, Nord) [m] & ({fmt(e.center[0])}, {fmt(e.center[1])}) \\",
                rf"Semiassi [m] & {fmt(e.semi_major_m)} / {fmt(e.semi_minor_m)} \\",
                rf"Orientazione semiasse maggiore [$^\circ$ da Est] & {fmt(e.angle_deg)} \\",
                rf"Fattore $k$ ({latex_escape(e.kind)}, {fmt(e.level, 0, pct=True)}) & {fmt(e.k, 4)} \\",
                rf"Copertura empirica & {fmt(e.empirical_coverage, 1, pct=True)} \\",
            ]
        lines.append(r"\bottomrule\end{tabular}\end{table}")
        if e is None:
            lines.append(rf"Ellisse non calcolabile: {latex_escape(land.ellipse_error)}.")
        elif not e.gaussian_consistent:
            lines.append(
                r"\textbf{Attenzione:} la copertura empirica dell'ellisse si discosta dal livello nominale "
                rf"oltre la tolleranza ($\pm$\,{fmt(e.coverage_tolerance, 1, pct=True)}): la distribuzione degli "
                r"impatti non è gaussiana. Usare R95 e i quantili di distanza, non l'ellisse, per l'area di "
                r"sicurezza."
            )
    for key, caption in (
        ("dispersion", "Punti di impatto ed ellisse di predizione."),
        ("convergence", "Convergenza della media dell'apogeo al crescere del numero di voli."),
    ):
        if key in figures:
            lines += [
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.8\linewidth]{{{figures[key]}}}",
                rf"\caption{{{latex_escape(caption)}}}",
                r"\end{figure}",
            ]
    lines += [
        r"\section{Modello stocastico}",
        r"\begin{table}[H]\centering\begin{tabular}{ll}\toprule",
        r"Parametro & Distribuzione \\ \midrule",
    ]
    for name, spec in metadata.get("config", {}).get("stochastic", {}).items():
        desc = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in spec.items())
        lines.append(rf"\texttt{{{latex_escape(name)}}} & {latex_escape(desc)} \\")
    lines += [r"\bottomrule\end{tabular}\end{table}", r"\end{document}", ""]
    return "\n".join(lines)


def write_report(run_dir: Path, summary: RunSummary, metadata: dict[str, Any], figures: dict[str, str]) -> Path:
    path = run_dir / "report.tex"
    atomic_write_text(path, render_report(summary, metadata, figures))
    return path
