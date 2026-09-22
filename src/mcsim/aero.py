"""Stima semi-empirica delle curve Cd(Mach) del vettore.

Il coefficiente di resistenza è costruito per componenti, con riferimento
all'area frontale del corpo ``S_ref = π R²``:

1. **Attrito viscoso** — lastra piana turbolenta secondo Prandtl–Schlichting,
   ``Cf = 1 / (1.50 ln Re − 5.6)²``, con correzione di compressibilità
   ``(1 − 0.1 M²)`` e fattori di forma per corpo (fineness) e alette (spessore).
2. **Resistenza di base** — correlazione di Hoerner ``Cd_base = 0.12 + 0.13 M²``
   riferita all'area di base effettiva (ridotta dal boattail). A motore acceso il
   getto riempie la scia e la resistenza di base si annulla: è la differenza
   fisica tra curva *power-on* e *power-off*.
3. **Salita transonica** — moltiplicatore empirico sulle componenti di attrito e
   pressione: unitario fino a M = 0,80, +60 % a M = 1,05, poi decadimento lento,
   coerente con i dati pubblicati per razzi sperimentali snelli.
4. **Protuberanze** — fattore 1,30 su attrito e pressione per giunzioni, rail
   buttons e rugosità superficiale (pratica standard 1,2–1,4).
5. **Compressibilità subsonica** — correzione di Prandtl–Glauert ``1/√(1 − M²)``
   sulla componente di pressione, limitata a M = 0,8.

Il modello è deterministico e riproducibile: ``python -m mcsim.aero`` rigenera i
CSV in ``data/`` e il test ``test_aero.py`` verifica che i file su disco
coincidano con il modello. Per un progetto di volo reale queste curve vanno
sostituite con RASAero II, OpenRocket o CFD: basta puntare il TOML ai nuovi CSV.

Riferimenti: S. F. Hoerner, *Fluid-Dynamic Drag* (1965), cap. 3 e 13;
J. S. Barrowman, *The Practical Calculation of the Aerodynamic Characteristics of
Slender Finned Vehicles* (1967).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

SPEED_OF_SOUND_MS = 336.0
"""Velocità del suono a ~3 km di quota (ISA, -5 °C): quota rappresentativa della salita."""
KINEMATIC_VISCOSITY_M2S = 1.6e-5
"""Viscosità cinematica dell'aria alla stessa quota di riferimento."""
TRANSONIC_START_MACH = 0.80
TRANSONIC_PEAK_MACH = 1.05
TRANSONIC_RISE = 0.60
"""Aumento relativo di resistenza tra M = 0,80 e M = 1,05."""
EXCRESCENCE_FACTOR = 1.30
"""Margine per protuberanze, giunzioni, rail buttons e rugosità: pratica standard 1,2–1,4."""


@dataclass(frozen=True, slots=True)
class Airframe:
    """Geometria minima necessaria alla stima aerodinamica (tutto in metri)."""

    body_radius_m: float
    body_length_m: float
    nose_length_m: float
    base_radius_m: float
    fin_count: int
    fin_root_chord_m: float
    fin_tip_chord_m: float
    fin_span_m: float
    fin_thickness_m: float

    @property
    def reference_area_m2(self) -> float:
        return math.pi * self.body_radius_m**2

    @property
    def fineness_ratio(self) -> float:
        return self.body_length_m / (2.0 * self.body_radius_m)

    @property
    def body_wetted_area_m2(self) -> float:
        """Ogiva (fattore 0,85 sulla superficie del cilindro equivalente) più tubo."""
        d = 2.0 * self.body_radius_m
        return math.pi * d * (0.85 * self.nose_length_m + (self.body_length_m - self.nose_length_m))

    @property
    def fins_wetted_area_m2(self) -> float:
        planform = 0.5 * (self.fin_root_chord_m + self.fin_tip_chord_m) * self.fin_span_m
        return 2.0 * self.fin_count * planform

    @property
    def mean_fin_chord_m(self) -> float:
        return 0.5 * (self.fin_root_chord_m + self.fin_tip_chord_m)


def _skin_friction_coefficient(reynolds: FloatArray) -> FloatArray:
    """Prandtl–Schlichting per lastra piana turbolenta, con limite laminare a Re bassi."""
    re = np.maximum(reynolds, 1.0e4)
    return 1.0 / (1.50 * np.log(re) - 5.6) ** 2


def _transonic_factor(mach: FloatArray) -> FloatArray:
    """1 fino a M = 0,80, +60 % a M = 1,05, poi decadimento ~M^-0,3."""
    factor = np.ones_like(mach)
    rise = (mach > TRANSONIC_START_MACH) & (mach <= TRANSONIC_PEAK_MACH)
    x = (mach[rise] - TRANSONIC_START_MACH) / (TRANSONIC_PEAK_MACH - TRANSONIC_START_MACH)
    factor[rise] = 1.0 + TRANSONIC_RISE * x**2
    supersonic = mach > TRANSONIC_PEAK_MACH
    factor[supersonic] = (1.0 + TRANSONIC_RISE) * (TRANSONIC_PEAK_MACH / mach[supersonic]) ** 0.3
    return factor


def drag_curves(airframe: Airframe, mach: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Restituisce ``(cd_power_off, cd_power_on)`` per i numeri di Mach richiesti."""
    mach = np.asarray(mach, dtype=float)
    if np.any(mach <= 0):
        raise ValueError("Il numero di Mach deve essere positivo")
    speed = mach * SPEED_OF_SOUND_MS
    s_ref = airframe.reference_area_m2

    # 1. Attrito viscoso su corpo e alette.
    re_body = speed * airframe.body_length_m / KINEMATIC_VISCOSITY_M2S
    re_fins = speed * airframe.mean_fin_chord_m / KINEMATIC_VISCOSITY_M2S
    compressibility = np.maximum(1.0 - 0.1 * mach**2, 0.5)
    cf_body = _skin_friction_coefficient(re_body) * compressibility
    cf_fins = _skin_friction_coefficient(re_fins) * compressibility
    form_body = 1.0 + 1.0 / (2.0 * airframe.fineness_ratio)
    form_fins = 1.0 + 2.0 * airframe.fin_thickness_m / airframe.mean_fin_chord_m
    cd_friction = (
        cf_body * form_body * airframe.body_wetted_area_m2 + cf_fins * form_fins * airframe.fins_wetted_area_m2
    ) / s_ref

    # 2. Pressione su ogiva e bordi d'attacco (piccola, corretta per compressibilità).
    prandtl_glauert = 1.0 / np.sqrt(np.maximum(1.0 - np.minimum(mach, 0.8) ** 2, 1.0e-3))
    cd_nose = 0.008 * prandtl_glauert
    cd_fin_edges = airframe.fin_count * airframe.fin_thickness_m * airframe.fin_span_m / s_ref * 0.12 * prandtl_glauert

    # 3. Resistenza di base: presente solo a motore spento.
    base_area_ratio = (airframe.base_radius_m / airframe.body_radius_m) ** 2
    cd_base = (0.12 + 0.13 * mach**2) * base_area_ratio

    transonic = _transonic_factor(mach)
    cd_clean = (cd_friction + cd_nose + cd_fin_edges) * EXCRESCENCE_FACTOR * transonic
    cd_power_off = cd_clean + cd_base * transonic
    cd_power_on = cd_clean
    return cd_power_off, cd_power_on


def mach_grid(start: float = 0.01, stop: float = 1.50, step: float = 0.01) -> FloatArray:
    return np.round(np.arange(start, stop + 0.5 * step, step), 4)


def write_curves(airframe: Airframe, data_dir: Path) -> tuple[Path, Path]:
    """Scrive ``cd_power_off.csv`` e ``cd_power_on.csv`` (due colonne: Mach, Cd)."""
    mach = mach_grid()
    cd_off, cd_on = drag_curves(airframe, mach)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = (data_dir / "cd_power_off.csv", data_dir / "cd_power_on.csv")
    for path, cd in zip(paths, (cd_off, cd_on), strict=True):
        rows = "\n".join(f"{m:.4f},{c:.6f}" for m, c in zip(mach, cd, strict=True))
        path.write_text(rows + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - utilità da riga di comando
    from .config import DATA_DIR, RocketConfig, airframe_from_config

    parser = argparse.ArgumentParser(description="Rigenera le curve Cd(Mach) in data/.")
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    off, on = write_curves(airframe_from_config(RocketConfig()), args.out)
    mach = np.array([0.1, 0.3, 0.6, 0.8, 1.0, 1.05, 1.5])
    cd_off, cd_on = drag_curves(airframe_from_config(RocketConfig()), mach)
    print(f"Scritti {off.name} e {on.name} in {args.out}")
    for m, a, b in zip(mach, cd_off, cd_on, strict=True):
        print(f"  M {m:4.2f}   Cd_off {a:.3f}   Cd_on {b:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
