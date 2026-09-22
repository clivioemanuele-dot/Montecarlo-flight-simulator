"""Configurazione tipizzata del simulatore: unica fonte di verità del modello.

I valori di default descrivono un vettore realmente volato e pubblicamente
documentato, così che il simulatore parta con dati coerenti invece che inventati:

- **Vettore**: *Calisto*, il razzo sperimentale del team universitario Projeto
  Jupiter usato come caso di riferimento nella documentazione di RocketPy
  (geometria completa: ogiva von Kármán, alette trapezoidali con cant di 0,5°,
  boattail, rail buttons, recupero dual-deploy).
- **Motore**: Cesaroni Pro75 **M1670 Blue Streak** (5 grani BATES), dati di
  certificazione pubblici: impulso totale 6026,35 N·s, spinta media 1545,22 N,
  spinta di picco 2200 N, tempo di combustione 3,9 s, massa di propellente
  2,956 kg, massa totale 5,231 kg (da cui massa a secco 2,275 kg).
- **Aerodinamica**: curve Cd(Mach) power-off e power-on in ``data/``, calcolate
  da :mod:`mcsim.aero` sulla geometria reale del vettore.

Tutte le grandezze sono SI e l'unità è nel nome del campo (``_m``, ``_kg``,
``_s``, ``_deg``, ``_ms``, ``_m2``): un errore di unità diventa visibile in review.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from .aero import Airframe
from .distributions import Dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    """Cartella ``data/``: accanto al pacchetto (installazione editabile) o nella cartella di lavoro."""
    for candidate in (PROJECT_ROOT / "data", Path.cwd() / "data"):
        if candidate.is_dir():
            return candidate
    return PROJECT_ROOT / "data"


DATA_DIR = _default_data_dir()
"""Curve di spinta e di resistenza. Sostituibili da TOML con percorsi propri."""


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Sito di lancio e parametri del profilo di vento.

    Il default è Torino (sede del progetto): sostituire con le coordinate del
    campo di lancio reale, per esempio Santa Margarida (EuRoC) o Roccaraso.
    """

    latitude_deg: float = 45.06
    longitude_deg: float = 7.66
    elevation_m: float = 250.0
    wind_ref_height_m: float = 10.0
    """Quota AGL a cui è riferita la velocità del vento campionata (anemometro standard)."""
    wind_profile_top_m: float = 6000.0
    """Quota AGL massima del profilo tabulato; oltre, RocketPy estrapola costante."""


@dataclass(frozen=True, slots=True)
class MotorConfig:
    """Cesaroni Pro75 M1670 Blue Streak (``nozzle_to_combustion_chamber``: +x verso la camera)."""

    thrust_curve_csv: Path = DATA_DIR / "thrust_curve_M1670.csv"
    burn_time_s: float = 3.9
    dry_mass_kg: float = 2.275
    """Massa totale certificata 5,231 kg meno il propellente 2,956 kg."""
    dry_inertia_kgm2: tuple[float, float, float] = (0.157, 0.157, 0.0025)
    """Inerzie del caso Pro75-5G, scalate sulla massa a secco reale."""
    nozzle_radius_m: float = 0.033
    throat_radius_m: float = 0.011
    grain_number: int = 5
    grain_density_kgm3: float = 1815.0
    grain_outer_radius_m: float = 0.033
    grain_initial_inner_radius_m: float = 0.015
    grain_initial_height_m: float = 0.120
    grain_separation_m: float = 0.005
    """Geometria BATES che riproduce i 2,956 kg di propellente certificati."""
    grains_center_of_mass_position_m: float = 0.397
    center_of_dry_mass_position_m: float = 0.317


@dataclass(frozen=True, slots=True)
class ParachuteConfig:
    """Paracadute. ``trigger``: quota AGL [m] in discesa, oppure ``"apogee"``."""

    name: str
    cd_s_m2: float
    trigger: float | Literal["apogee"]
    lag_s: float
    sampling_rate_hz: float = 105.0
    noise_pa: tuple[float, float, float] = (0.0, 8.3, 0.5)
    """(bias, deviazione standard, correlazione) del rumore del sensore di pressione."""


@dataclass(frozen=True, slots=True)
class TailConfig:
    """Boattail terminale: riduce l'area di base e quindi la resistenza."""

    top_radius_m: float = 0.0635
    bottom_radius_m: float = 0.0435
    length_m: float = 0.060
    position_m: float = -1.194656


@dataclass(frozen=True, slots=True)
class RailButtonsConfig:
    """Pattini di guida: determinano la lunghezza efficace di rampa."""

    upper_position_m: float = 0.0818
    lower_position_m: float = -0.6182
    angular_position_deg: float = 45.0


@dataclass(frozen=True, slots=True)
class RocketConfig:
    """Vettore in sistema ``tail_to_nose``, origine nel centro di massa senza motore."""

    radius_m: float = 0.0635
    mass_kg: float = 14.426
    """Massa a secco senza motore."""
    inertia_kgm2: tuple[float, float, float] = (6.321, 6.321, 0.034)
    power_off_drag_csv: Path = DATA_DIR / "cd_power_off.csv"
    power_on_drag_csv: Path = DATA_DIR / "cd_power_on.csv"
    motor_position_m: float = -1.255
    nose_length_m: float = 0.55829
    nose_kind: str = "vonKarman"
    nose_position_m: float = 1.278
    fin_n: int = 4
    fin_root_chord_m: float = 0.120
    fin_tip_chord_m: float = 0.060
    fin_span_m: float = 0.110
    fin_position_m: float = -1.04956
    fin_cant_angle_deg: float = 0.5
    """Cant reale delle alette: induce rollio, quindi il volo usa davvero i 6 gradi di libertà."""
    fin_sweep_length_m: float | None = None
    fin_thickness_m: float = 0.004
    """Pannelli in G10 da 4 mm: usato solo dalla stima aerodinamica."""
    tail: TailConfig | None = field(default_factory=TailConfig)
    rail_buttons: RailButtonsConfig | None = field(default_factory=RailButtonsConfig)
    drogue: ParachuteConfig | None = field(
        default_factory=lambda: ParachuteConfig("Drogue", cd_s_m2=1.0, trigger="apogee", lag_s=1.5)
    )
    main: ParachuteConfig = field(
        default_factory=lambda: ParachuteConfig("Main", cd_s_m2=10.0, trigger=800.0, lag_s=1.5)
    )


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    """Rampa e limiti di integrazione."""

    rail_length_m: float = 5.2
    inclination_deg: float = 85.0
    heading_deg: float = 0.0
    max_time_s: float = 600.0
    """Tempo simulato massimo: un volo che non atterra entro questo limite è ``NOT_LANDED``."""


def airframe_from_config(rocket: RocketConfig) -> Airframe:
    """Geometria per la stima aerodinamica, derivata dalla configurazione del vettore."""
    return Airframe(
        body_radius_m=rocket.radius_m,
        body_length_m=rocket.nose_position_m - rocket.motor_position_m,
        nose_length_m=rocket.nose_length_m,
        base_radius_m=rocket.tail.bottom_radius_m if rocket.tail else rocket.radius_m,
        fin_count=rocket.fin_n,
        fin_root_chord_m=rocket.fin_root_chord_m,
        fin_tip_chord_m=rocket.fin_tip_chord_m,
        fin_span_m=rocket.fin_span_m,
        fin_thickness_m=rocket.fin_thickness_m,
    )


def default_stochastic_model() -> dict[str, Dist]:
    """Modello di incertezza di riferimento (le chiavi sono campi di ``FlightInputs``).

    Gli ordini di grandezza seguono la pratica industriale per un razzo
    sperimentale di classe M e **vanno confermati** con i dati del tuo vettore:
    pesate reali (massa), certificazione e temperatura del lotto (impulso, tempo
    di combustione), CFD o galleria (Cd), prove di apertura (CdS, ritardi),
    climatologia o previsione ensemble del sito (vento).
    """
    return {
        # Rampa regolata con inclinometro; troncata ben sotto i 90° (P1-10).
        "inclination_deg": Dist.truncnorm(mean=85.0, std=0.5, low=83.0, high=87.0),
        # Azimut di rampa: tolleranza di puntamento, NON U(0°, 360°) (P1-02).
        "heading_deg": Dist.normal(mean=0.0, std=1.0),
        # Massa a secco: tolleranza di pesata e integrazione, 0,5 % (1σ).
        "rocket_mass_kg": Dist.truncnorm_sigma(mean=14.426, std=0.072),
        # Impulso totale: dispersione di lotto dichiarata dai costruttori, ~2 % (1σ).
        "impulse_scale": Dist.truncnorm_sigma(mean=1.0, std=0.02),
        # Temperatura del grano: tempo di combustione ±1,5 % a impulso costante.
        "burn_time_scale": Dist.truncnorm_sigma(mean=1.0, std=0.015),
        # Fattori moltiplicativi strettamente positivi: lognormali.
        "drag_scale": Dist.lognormal(median=1.0, sigma=0.08),
        "drogue_cds_scale": Dist.lognormal(median=1.0, sigma=0.08),
        "main_cds_scale": Dist.lognormal(median=1.0, sigma=0.08),
        "drogue_lag_s": Dist.lognormal(median=1.5, sigma=0.2),
        "main_lag_s": Dist.lognormal(median=1.5, sigma=0.2),
        # Vento a 10 m AGL: Weibull (k = 2 ≡ Rayleigh), direzione di provenienza von Mises.
        "wind_speed_ref_ms": Dist.weibull(shape=2.0, scale=4.0),
        "wind_from_deg": Dist.vonmises_deg(mean=270.0, kappa=3.0),
        # Esponente della legge di potenza del profilo verticale (terreno aperto).
        "wind_shear_exponent": Dist.truncnorm(mean=0.14, std=0.04, low=0.06, high=0.30),
    }


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Configurazione completa di una campagna Monte Carlo."""

    site: SiteConfig = field(default_factory=SiteConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    rocket: RocketConfig = field(default_factory=RocketConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    stochastic: dict[str, Dist] = field(default_factory=default_stochastic_model)

    def to_dict(self) -> dict[str, Any]:
        """Forma serializzabile (JSON) per i metadati del run."""
        out: dict[str, Any] = {
            "site": asdict(self.site),
            "motor": asdict(self.motor),
            "rocket": asdict(self.rocket),
            "launch": asdict(self.launch),
        }
        out["stochastic"] = {name: dist.to_dict() for name, dist in self.stochastic.items()}
        return out

    def check_data_files(self) -> None:
        """Verifica subito che curve di spinta e di resistenza esistano."""
        missing = [
            path
            for path in (self.motor.thrust_curve_csv, self.rocket.power_off_drag_csv, self.rocket.power_on_drag_csv)
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "File di dati mancanti: " + ", ".join(str(p) for p in missing) + ". Rigenera le curve con "
                "`python -m mcsim.aero` o correggi i percorsi nel TOML."
            )


# ---------------------------------------------------------------------- caricamento TOML
def _to_tuple(value: Any) -> Any:
    return tuple(_to_tuple(v) for v in value) if isinstance(value, list) else value


def _convert(current: Any, value: Any) -> Any:
    if isinstance(current, Path):
        return Path(value)
    return _to_tuple(value)


def _merge(instance: Any, overrides: dict[str, Any], path: str) -> Any:
    """Applica ``overrides`` a un dataclass frozen, rifiutando chiavi sconosciute."""
    valid = {f.name for f in fields(instance)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid:
            raise KeyError(f"Chiave di configurazione sconosciuta: {path}{key}")
        current = getattr(instance, key)
        if value is False and key in ("drogue", "tail", "rail_buttons"):
            changes[key] = None
        elif is_dataclass(current) and isinstance(value, dict):
            changes[key] = _merge(current, value, f"{path}{key}.")
        elif key in ("drogue", "main") and isinstance(value, dict):
            changes[key] = ParachuteConfig(**{k: _to_tuple(v) for k, v in value.items()})
        elif key == "tail" and isinstance(value, dict):
            changes[key] = TailConfig(**value)
        elif key == "rail_buttons" and isinstance(value, dict):
            changes[key] = RailButtonsConfig(**value)
        else:
            changes[key] = _convert(current, value)
    return replace(instance, **changes)


def _resolve_paths(cfg: SimConfig, base: Path) -> SimConfig:
    """I percorsi relativi nel TOML si riferiscono alla cartella del TOML stesso."""
    motor = replace(cfg.motor, thrust_curve_csv=_abs(cfg.motor.thrust_curve_csv, base))
    rocket = replace(
        cfg.rocket,
        power_off_drag_csv=_abs(cfg.rocket.power_off_drag_csv, base),
        power_on_drag_csv=_abs(cfg.rocket.power_on_drag_csv, base),
    )
    return replace(cfg, motor=motor, rocket=rocket)


def _abs(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path | str) -> SimConfig:
    """Carica un file TOML sopra i default.

    Le sezioni ``[site]``, ``[motor]``, ``[rocket]``, ``[launch]`` sovrascrivono
    campo per campo. La sezione ``[stochastic]``, se presente, **sostituisce
    interamente** il modello di incertezza (niente parametri ereditati di nascosto).
    In ``[rocket]``, ``drogue = false``, ``tail = false`` o ``rail_buttons = false``
    rimuovono il componente.
    """
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    stochastic = data.pop("stochastic", None)
    cfg = cast(SimConfig, _merge(SimConfig(), data, ""))
    if stochastic is not None:
        cfg = replace(cfg, stochastic={name: Dist.from_dict(spec) for name, spec in stochastic.items()})
    return _resolve_paths(cfg, config_path.parent)
