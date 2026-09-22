"""Factory pure del modello di volo.

Ogni volo costruisce **i propri** oggetti ``Environment``, ``SolidMotor`` e
``Rocket`` (costo misurato ~20 ms contro ~900 ms di integrazione). Il codice
originale riusava lo stesso ``Rocket`` per tutti i voli: i ``Parachute`` di
RocketPy accumulano i segnali di pressione a ogni volo (≈1,4 MB trattenuti per
volo) e il rumore AR(1) del sensore riparte dall'ultimo campione del volo
precedente, rendendo le iterazioni dipendenti tra loro (P1-05).

Le curve di spinta e di resistenza sono lette una sola volta per processo
(``lru_cache``) e poi scalate in memoria dai fattori campionati.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from rocketpy import Environment, Flight, Function, Rocket, SolidMotor

from . import _compat  # noqa: F401  (deve precedere l'uso di RocketPy)
from .config import MotorConfig, ParachuteConfig, RocketConfig, SimConfig, SiteConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FlightInputs:
    """Valori di tutti i parametri perturbabili per un singolo volo."""

    inclination_deg: float
    heading_deg: float
    rocket_mass_kg: float
    impulse_scale: float
    burn_time_scale: float
    drag_scale: float
    drogue_cds_scale: float
    main_cds_scale: float
    drogue_lag_s: float
    main_lag_s: float
    wind_speed_ref_ms: float
    wind_from_deg: float
    wind_shear_exponent: float

    @classmethod
    def nominal(cls, cfg: SimConfig) -> Self:
        """Volo nominale: nessuna perturbazione, vento nullo."""
        drogue = cfg.rocket.drogue
        return cls(
            inclination_deg=cfg.launch.inclination_deg,
            heading_deg=cfg.launch.heading_deg,
            rocket_mass_kg=cfg.rocket.mass_kg,
            impulse_scale=1.0,
            burn_time_scale=1.0,
            drag_scale=1.0,
            drogue_cds_scale=1.0,
            main_cds_scale=1.0,
            drogue_lag_s=drogue.lag_s if drogue is not None else 0.0,
            main_lag_s=cfg.rocket.main.lag_s,
            wind_speed_ref_ms=0.0,
            wind_from_deg=0.0,
            wind_shear_exponent=0.14,
        )


@lru_cache(maxsize=8)
def load_curve(path: Path) -> FloatArray:
    """Carica una curva a due colonne (CSV senza intestazione), una volta per processo."""
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError(f"{path}: attese due colonne (ascissa, valore) e almeno due righe")
    return np.ascontiguousarray(data)


def wind_profile(site: SiteConfig, inputs: FlightInputs) -> tuple[FloatArray, FloatArray]:
    """Profilo di vento a legge di potenza, direzione costante con la quota.

    ``V(z) = V_ref · (z / z_ref)^α`` con ``z`` quota AGL (limitata inferiormente a 1 m).
    Convenzione meteorologica: ``wind_from_deg`` è la direzione di *provenienza*,
    da Nord in senso orario; quindi ``u = -V sin θ`` (Est), ``v = -V cos θ`` (Nord).

    Returns
    -------
    wind_u, wind_v
        Tabelle ``(quota ASL [m], componente [m/s])`` per ``custom_atmosphere``.
    """
    z_agl = np.concatenate(([0.0], np.geomspace(1.0, site.wind_profile_top_m, 60)))
    speed = inputs.wind_speed_ref_ms * (np.maximum(z_agl, 1.0) / site.wind_ref_height_m) ** inputs.wind_shear_exponent
    theta = math.radians(inputs.wind_from_deg)
    h_asl = site.elevation_m + z_agl
    return np.column_stack((h_asl, -speed * math.sin(theta))), np.column_stack((h_asl, -speed * math.cos(theta)))


def build_environment(site: SiteConfig, inputs: FlightInputs) -> Environment:
    """Atmosfera standard ISA per pressione e temperatura, vento dal profilo campionato.

    Nota: con ``standard_atmosphere`` (codice originale) il vento è identicamente
    nullo e la data impostata con ``set_date`` non ha alcun effetto (P1-01).
    """
    env = Environment(latitude=site.latitude_deg, longitude=site.longitude_deg, elevation=site.elevation_m)
    wind_u, wind_v = wind_profile(site, inputs)
    env.set_atmospheric_model(type="custom_atmosphere", wind_u=wind_u, wind_v=wind_v)
    return env


def build_motor(motor: MotorConfig, inputs: FlightInputs) -> SolidMotor:
    """Motore reale con dispersione di impulso e di tempo di combustione.

    ``burn_time_scale = k`` dilata l'asse dei tempi e divide la spinta per ``k``:
    l'impulso totale resta costante, come per la variazione di velocità di
    combustione con la temperatura del grano. ``impulse_scale`` scala la spinta a
    parità di propellente, cioè modella la dispersione di lotto dell'impulso.
    """
    k_t, k_i = inputs.burn_time_scale, inputs.impulse_scale
    curve = load_curve(motor.thrust_curve_csv).copy()
    curve[:, 0] *= k_t
    curve[:, 1] *= k_i / k_t
    return SolidMotor(
        thrust_source=curve,
        burn_time=motor.burn_time_s * k_t,
        dry_mass=motor.dry_mass_kg,
        dry_inertia=motor.dry_inertia_kgm2,
        nozzle_radius=motor.nozzle_radius_m,
        throat_radius=motor.throat_radius_m,
        grain_number=motor.grain_number,
        grain_density=motor.grain_density_kgm3,
        grain_outer_radius=motor.grain_outer_radius_m,
        grain_initial_inner_radius=motor.grain_initial_inner_radius_m,
        grain_initial_height=motor.grain_initial_height_m,
        grain_separation=motor.grain_separation_m,
        grains_center_of_mass_position=motor.grains_center_of_mass_position_m,
        center_of_dry_mass_position=motor.center_of_dry_mass_position_m,
        nozzle_position=0.0,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


def _scaled_drag(path: Path, scale: float) -> Function:
    """Curva Cd(Mach) moltiplicata per il fattore di incertezza campionato.

    RocketPy accetta un percorso CSV o una ``Function``: si costruisce la
    ``Function`` in memoria per non rileggere il file a ogni volo e per poter
    applicare il fattore di incertezza senza scrivere file temporanei.
    """
    curve = load_curve(path).copy()
    curve[:, 1] *= scale
    return Function(curve, "Mach Number", "Drag Coefficient", "linear", "constant")


def _add_parachute(rocket: Rocket, chute: ParachuteConfig, cds_scale: float, lag_s: float) -> None:
    rocket.add_parachute(
        name=chute.name,
        cd_s=chute.cd_s_m2 * cds_scale,
        trigger=chute.trigger,
        sampling_rate=chute.sampling_rate_hz,
        lag=lag_s,
        noise=chute.noise_pa,
    )


def build_rocket(cfg: RocketConfig, motor: SolidMotor, inputs: FlightInputs) -> Rocket:
    """Vettore completo: ogiva, alette con cant, boattail, rail buttons, recupero.

    L'inerzia scala con la massa (stessa distribuzione geometrica).
    """
    mass_ratio = inputs.rocket_mass_kg / cfg.mass_kg
    rocket = Rocket(
        radius=cfg.radius_m,
        mass=inputs.rocket_mass_kg,
        inertia=tuple(i * mass_ratio for i in cfg.inertia_kgm2),
        power_off_drag=_scaled_drag(cfg.power_off_drag_csv, inputs.drag_scale),
        power_on_drag=_scaled_drag(cfg.power_on_drag_csv, inputs.drag_scale),
        center_of_mass_without_motor=0.0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=cfg.motor_position_m)
    rocket.add_nose(length=cfg.nose_length_m, kind=cfg.nose_kind, position=cfg.nose_position_m)
    rocket.add_trapezoidal_fins(
        n=cfg.fin_n,
        root_chord=cfg.fin_root_chord_m,
        tip_chord=cfg.fin_tip_chord_m,
        span=cfg.fin_span_m,
        position=cfg.fin_position_m,
        cant_angle=cfg.fin_cant_angle_deg,
        sweep_length=cfg.fin_sweep_length_m,
    )
    if cfg.tail is not None:
        rocket.add_tail(
            top_radius=cfg.tail.top_radius_m,
            bottom_radius=cfg.tail.bottom_radius_m,
            length=cfg.tail.length_m,
            position=cfg.tail.position_m,
        )
    if cfg.rail_buttons is not None:
        rocket.set_rail_buttons(
            upper_button_position=cfg.rail_buttons.upper_position_m,
            lower_button_position=cfg.rail_buttons.lower_position_m,
            angular_position=cfg.rail_buttons.angular_position_deg,
        )
    if cfg.drogue is not None:
        _add_parachute(rocket, cfg.drogue, inputs.drogue_cds_scale, inputs.drogue_lag_s)
    _add_parachute(rocket, cfg.main, inputs.main_cds_scale, inputs.main_lag_s)
    return rocket


def build_flight(cfg: SimConfig, inputs: FlightInputs) -> Flight:
    """Costruisce modello e integra il volo 6-DOF (può sollevare eccezioni: le gestisce il runner)."""
    env = build_environment(cfg.site, inputs)
    motor = build_motor(cfg.motor, inputs)
    rocket = build_rocket(cfg.rocket, motor, inputs)
    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=cfg.launch.rail_length_m,
        inclination=inputs.inclination_deg,
        heading=inputs.heading_deg,
        max_time=cfg.launch.max_time_s,
        terminate_on_apogee=False,
        verbose=False,
    )
