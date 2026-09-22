"""Distribuzioni di probabilità per i parametri incerti del volo.

Regole di modellazione applicate (vedi review, P1-09 e P1-10):

- grandezze **strettamente positive** (ritardi, fattori di scala su Cd e CdS,
  velocità del vento) non usano mai una normale non troncata, che genera valori
  negativi non fisici: si usano ``lognormal`` (fattori moltiplicativi) o
  ``weibull`` (velocità del vento);
- grandezze **limitate** (inclinazione della rampa < 90°, tolleranze di massa e
  impulso) usano ``truncnorm`` con limiti espliciti;
- **direzioni** usano ``vonmises_deg``, la distribuzione naturale su un cerchio,
  invece di una normale che ignora la periodicità a 360°.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

import numpy as np
from scipy import stats

DistKind = Literal["fixed", "normal", "truncnorm", "lognormal", "uniform", "vonmises_deg", "weibull"]


@dataclass(frozen=True, slots=True)
class Dist:
    """Distribuzione univariata di un parametro incerto.

    Usare i costruttori di classe (:meth:`truncnorm`, :meth:`lognormal`, …): i campi
    richiesti dipendono da ``kind`` e sono validati in ``__post_init__``.

    Attributes
    ----------
    kind
        Famiglia della distribuzione.
    mean, std
        Media e deviazione standard (``normal``, ``truncnorm``); per ``vonmises_deg``
        ``mean`` è la direzione media in gradi.
    low, high
        Limiti assoluti (``truncnorm``, ``uniform``).
    median, sigma
        Mediana e deviazione standard del logaritmo (``lognormal``).
    kappa
        Concentrazione della von Mises (0 = uniforme sul cerchio).
    shape, scale
        Parametri ``k`` e ``λ`` della Weibull.
    value
        Valore deterministico (``fixed``).
    """

    kind: DistKind
    mean: float | None = None
    std: float | None = None
    low: float | None = None
    high: float | None = None
    median: float | None = None
    sigma: float | None = None
    kappa: float | None = None
    shape: float | None = None
    scale: float | None = None
    value: float | None = None

    _REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "fixed": ("value",),
        "normal": ("mean", "std"),
        "truncnorm": ("mean", "std", "low", "high"),
        "lognormal": ("median", "sigma"),
        "uniform": ("low", "high"),
        "vonmises_deg": ("mean", "kappa"),
        "weibull": ("shape", "scale"),
    }

    # ------------------------------------------------------------------ costruttori
    @classmethod
    def fixed(cls, value: float) -> Self:
        return cls("fixed", value=value)

    @classmethod
    def normal(cls, mean: float, std: float) -> Self:
        return cls("normal", mean=mean, std=std)

    @classmethod
    def truncnorm(cls, mean: float, std: float, low: float, high: float) -> Self:
        return cls("truncnorm", mean=mean, std=std, low=low, high=high)

    @classmethod
    def truncnorm_sigma(cls, mean: float, std: float, n_sigma: float = 3.0) -> Self:
        """Normale troncata simmetricamente a ``mean ± n_sigma·std``."""
        return cls.truncnorm(mean, std, mean - n_sigma * std, mean + n_sigma * std)

    @classmethod
    def lognormal(cls, median: float, sigma: float) -> Self:
        return cls("lognormal", median=median, sigma=sigma)

    @classmethod
    def uniform(cls, low: float, high: float) -> Self:
        return cls("uniform", low=low, high=high)

    @classmethod
    def vonmises_deg(cls, mean: float, kappa: float) -> Self:
        return cls("vonmises_deg", mean=mean, kappa=kappa)

    @classmethod
    def weibull(cls, shape: float, scale: float) -> Self:
        return cls("weibull", shape=shape, scale=scale)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Costruisce da un dizionario (es. una tabella TOML ``{kind = "...", ...}``)."""
        params = {k: float(v) for k, v in data.items() if k != "kind"}
        return cls(kind=data["kind"], **params)

    # ------------------------------------------------------------------ validazione
    def __post_init__(self) -> None:
        if self.kind not in self._REQUIRED:
            raise ValueError(f"Distribuzione sconosciuta: {self.kind!r}")
        missing = [name for name in self._REQUIRED[self.kind] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.kind}: parametri mancanti {missing}")
        for name in ("mean", "std", "low", "high", "median", "sigma", "kappa", "shape", "scale", "value"):
            val = getattr(self, name)
            if val is not None and not math.isfinite(val):
                raise ValueError(f"{self.kind}: {name} deve essere finito, ricevuto {val}")
        if self.std is not None and self.std <= 0:
            raise ValueError(f"{self.kind}: std deve essere > 0")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("lognormal: sigma deve essere > 0")
        if self.median is not None and self.median <= 0:
            raise ValueError("lognormal: median deve essere > 0")
        if self.kappa is not None and self.kappa < 0:
            raise ValueError("vonmises_deg: kappa deve essere >= 0")
        if self.kind == "weibull" and (self._p("shape") <= 0 or self._p("scale") <= 0):
            raise ValueError("weibull: shape e scale devono essere > 0")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise ValueError(f"{self.kind}: serve low < high")

    # ------------------------------------------------------------------ campionamento
    def _p(self, name: str) -> float:
        """Parametro obbligatorio per ``kind`` (garantito non ``None`` da ``__post_init__``)."""
        value = getattr(self, name)
        assert value is not None, f"{self.kind}: {name} mancante"
        return float(value)

    def sample(self, rng: np.random.Generator) -> float:
        """Estrae un campione usando **solo** il generatore passato (mai lo stato globale)."""
        p = self._p
        match self.kind:
            case "fixed":
                return p("value")
            case "normal":
                return float(rng.normal(p("mean"), p("std")))
            case "truncnorm":
                mean, std = p("mean"), p("std")
                a, b = (p("low") - mean) / std, (p("high") - mean) / std
                return float(stats.truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))
            case "lognormal":
                return p("median") * math.exp(p("sigma") * float(rng.standard_normal()))
            case "uniform":
                return float(rng.uniform(p("low"), p("high")))
            case "vonmises_deg":
                angle = float(rng.vonmises(math.radians(p("mean")), p("kappa")))
                return math.degrees(angle) % 360.0
            case "weibull":
                return p("scale") * float(rng.weibull(p("shape")))
        raise AssertionError(self.kind)  # pragma: no cover

    # ------------------------------------------------------------------ descrizione
    def describe(self) -> str:
        """Descrizione compatta e leggibile, usata in metadati e report."""
        match self.kind:
            case "fixed":
                return f"fisso = {self.value:g}"
            case "normal":
                return f"N(μ={self.mean:g}, σ={self.std:g})"
            case "truncnorm":
                return f"TN(μ={self.mean:g}, σ={self.std:g}; [{self.low:g}, {self.high:g}])"
            case "lognormal":
                return f"LogN(mediana={self.median:g}, σ_ln={self.sigma:g})"
            case "uniform":
                return f"U[{self.low:g}, {self.high:g}]"
            case "vonmises_deg":
                return f"vonMises(μ={self.mean:g}°, κ={self.kappa:g})"
            case "weibull":
                return f"Weibull(k={self.shape:g}, λ={self.scale:g})"
        raise AssertionError(self.kind)  # pragma: no cover

    def to_dict(self) -> dict[str, Any]:
        """Solo i campi valorizzati, in forma serializzabile JSON/TOML."""
        out: dict[str, Any] = {"kind": self.kind}
        for name in self._REQUIRED[self.kind]:
            out[name] = getattr(self, name)
        return out
