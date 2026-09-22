"""Statistica della campagna: stime con incertezza, regioni di confidenza, controlli di validità.

Correzioni rispetto all'originale (P1-11, P1-12):

- l'ellisse "kσ" in 2D copre 39,3 / 86,5 / 98,9 %, non 68 / 95 / 99,7 %:
  qui il fattore è derivato dal livello richiesto (χ² a 2 gdl, oppure la
  regione di *predizione* di Hotelling che tiene conto della stima di media e
  covarianza, rilevante per N piccolo);
- ogni ellisse riporta la **copertura empirica**: se differisce dal livello
  nominale oltre 3 errori standard binomiali, la distribuzione non è gaussiana
  e l'ellisse non va usata come area di sicurezza (si usano R95 / quantili);
- media con IC t di Student, quantili con IC distribution-free (statistiche
  d'ordine), proporzioni con IC di Clopper–Pearson; ``ddof=1`` ovunque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from .status import FlightStatus

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------- scalari
def quantile_ci(x: npt.ArrayLike, q: float, level: float = 0.95) -> tuple[float, float]:
    """IC distribution-free per il quantile ``q`` tramite statistiche d'ordine.

    ``[x_(l), x_(u)]`` con ``l, u`` scelti dalla Binomiale(n, q) così che
    ``P(x_(l) ≤ ξ_q ≤ x_(u)) ≥ level``. Se ``n`` è troppo piccolo per il livello
    richiesto, l'estremo mancante è ``±inf``.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    if n == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - level
    lo_rank = int(stats.binom.ppf(alpha / 2, n, q))  # rank 1-based: x_(lo_rank)
    hi_rank = int(stats.binom.ppf(1 - alpha / 2, n, q)) + 1
    lo = xs[lo_rank - 1] if lo_rank >= 1 else -math.inf
    hi = xs[hi_rank - 1] if hi_rank <= n else math.inf
    return (float(lo), float(hi))


def proportion_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """IC esatto di Clopper–Pearson per una proporzione (es. tasso di voli balistici)."""
    if n == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    n: int
    mean: float
    std: float
    sem: float
    mean_ci: tuple[float, float]
    p05: float
    p50: float
    p95: float
    p95_ci: tuple[float, float]
    minimum: float
    maximum: float


def summarize_scalar(x: npt.ArrayLike, level: float = 0.95) -> ScalarSummary:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        nan = math.nan
        return ScalarSummary(n, nan, nan, nan, (nan, nan), nan, nan, nan, (nan, nan), nan, nan)
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    sem = std / math.sqrt(n)
    t = float(stats.t.ppf(0.5 + level / 2, n - 1))
    p05, p50, p95 = (float(v) for v in np.quantile(a, [0.05, 0.50, 0.95]))
    return ScalarSummary(
        n,
        mean,
        std,
        sem,
        (mean - t * sem, mean + t * sem),
        p05,
        p50,
        p95,
        quantile_ci(a, 0.95, level),
        float(a.min()),
        float(a.max()),
    )


# ---------------------------------------------------------------------------- ellisse
@dataclass(frozen=True, slots=True)
class DispersionEllipse:
    level: float
    kind: Literal["prediction", "chi2"]
    n: int
    center: tuple[float, float]
    semi_major_m: float
    semi_minor_m: float
    angle_deg: float
    """Orientazione del semiasse maggiore, antioraria dall'asse Est."""
    k: float
    """Fattore di scala sulle deviazioni standard principali."""
    empirical_coverage: float
    coverage_tolerance: float

    @property
    def gaussian_consistent(self) -> bool:
        """Copertura empirica compatibile con il livello nominale (entro 3 errori standard)."""
        return abs(self.empirical_coverage - self.level) <= self.coverage_tolerance


def ellipse_scale(n: int, level: float, kind: Literal["prediction", "chi2"] = "prediction") -> float:
    """Fattore ``k`` dell'ellisse bivariata.

    - ``chi2``: popolazione nota, ``k² = χ²₂(level)`` (2,4477 per il 95 %).
    - ``prediction``: regione che contiene un **nuovo** impatto con probabilità
      ``level`` quando media e covarianza sono stimate da ``n`` campioni:
      ``k² = p(n−1)(n+1) / (n(n−p)) · F_{p, n−p}(level)``, con ``p = 2``.
      Per ``n = 1000`` coincide con χ² entro lo 0,3 %; per ``n = 10`` vale 3,32 invece di 2,45.
    """
    p = 2
    if kind == "chi2":
        return math.sqrt(float(stats.chi2.ppf(level, p)))
    if n <= p:
        raise ValueError(f"Servono almeno {p + 1} punti per la regione di predizione")
    k2 = p * (n - 1) * (n + 1) / (n * (n - p)) * float(stats.f.ppf(level, p, n - p))
    return math.sqrt(k2)


def dispersion_ellipse(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    level: float = 0.95,
    kind: Literal["prediction", "chi2"] = "prediction",
) -> DispersionEllipse:
    pts = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    n = len(pts)
    k = ellipse_scale(n, level, kind)
    center = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False, ddof=1)
    eigval, eigvec = np.linalg.eigh(cov)
    if eigval[0] <= 1e-12 * max(eigval[1], 1e-300):
        raise ValueError("Covarianza degenere: impatti allineati o coincidenti")
    order = eigval.argsort()[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    d = pts - center
    maha2 = np.einsum("ij,ij->i", d @ np.linalg.inv(cov), d)
    coverage = float(np.mean(maha2 <= k * k))
    return DispersionEllipse(
        level=level,
        kind=kind,
        n=n,
        center=(float(center[0]), float(center[1])),
        semi_major_m=float(k * math.sqrt(eigval[0])),
        semi_minor_m=float(k * math.sqrt(eigval[1])),
        angle_deg=float(math.degrees(math.atan2(eigvec[1, 0], eigvec[0, 0]))),
        k=k,
        empirical_coverage=coverage,
        coverage_tolerance=3.0 * math.sqrt(level * (1.0 - level) / n),
    )


# ---------------------------------------------------------------------------- campagna
@dataclass(frozen=True, slots=True)
class LandingSummary:
    n: int
    cep50_m: float
    """Raggio attorno al punto medio di impatto che contiene il 50 % degli impatti."""
    r95_m: float
    """Idem al 95 %: area di recupero distribution-free."""
    range_from_pad: ScalarSummary
    ellipse: DispersionEllipse | None
    ellipse_error: str


@dataclass(frozen=True, slots=True)
class RunSummary:
    n_total: int
    status_counts: dict[str, int]
    success_rate: float
    success_rate_ci: tuple[float, float]
    ballistic_rate: float
    ballistic_rate_ci: tuple[float, float]
    apogee: ScalarSummary
    max_mach: ScalarSummary
    main_deploy_speed: ScalarSummary
    static_margin: ScalarSummary
    landing_nominal: LandingSummary | None
    """Impatti con recupero nominale (status ``ok``)."""
    max_range_all_physical_m: float
    """Massima distanza dalla rampa su tutti i voli fisici, balistici inclusi."""


def landing_summary(x: FloatArray, y: FloatArray, level: float = 0.95) -> LandingSummary | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return None
    r_center = np.hypot(x - x.mean(), y - y.mean())
    try:
        ellipse: DispersionEllipse | None = dispersion_ellipse(x, y, level)
        err = ""
    except ValueError as exc:
        ellipse, err = None, str(exc)
    return LandingSummary(
        n=int(x.size),
        cep50_m=float(np.quantile(r_center, 0.50)),
        r95_m=float(np.quantile(r_center, 0.95)),
        range_from_pad=summarize_scalar(np.hypot(x, y), level),
        ellipse=ellipse,
        ellipse_error=err,
    )


def summarize_run(df: pd.DataFrame, level: float = 0.95) -> RunSummary:
    """Statistiche della campagna a partire da ``results.csv``.

    Le statistiche fisiche usano solo gli status ``ok`` e ``ballistic``; i
    fallimenti numerici sono contati e riportati, mai mescolati alle stime.
    """
    status = df["status"].astype(str)
    counts = {s.value: int((status == s.value).sum()) for s in FlightStatus}
    n = len(df)
    physical = df[status.isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])]
    ok = df[status == FlightStatus.OK.value]
    k_ok = counts[FlightStatus.OK.value]
    k_bal = counts[FlightStatus.BALLISTIC.value]
    n_phys = len(physical)
    r_all = np.hypot(physical["impact_x_m"].to_numpy(float), physical["impact_y_m"].to_numpy(float))
    return RunSummary(
        n_total=n,
        status_counts=counts,
        success_rate=k_ok / n if n else math.nan,
        success_rate_ci=proportion_ci(k_ok, n, level),
        ballistic_rate=k_bal / n_phys if n_phys else math.nan,
        ballistic_rate_ci=proportion_ci(k_bal, n_phys, level),
        apogee=summarize_scalar(physical["apogee_agl_m"], level),
        max_mach=summarize_scalar(physical["max_mach"], level),
        main_deploy_speed=summarize_scalar(ok["main_deploy_speed_ms"], level),
        static_margin=summarize_scalar(physical["static_margin_liftoff_cal"], level),
        landing_nominal=landing_summary(ok["impact_x_m"].to_numpy(float), ok["impact_y_m"].to_numpy(float), level),
        max_range_all_physical_m=float(np.max(r_all)) if r_all.size else math.nan,
    )


def running_mean_band(x: npt.ArrayLike, z: float = 1.96) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Media progressiva ± ``z``·SEM progressivo, per verificare la convergenza al crescere di N."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = np.arange(1, a.size + 1, dtype=float)
    mean = np.cumsum(a) / n
    sq = np.cumsum(a * a)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sq - n * mean * mean) / (n - 1)
        sem = np.sqrt(np.clip(var, 0.0, None) / n)
    sem[0] = np.nan
    return n, mean, z * sem
