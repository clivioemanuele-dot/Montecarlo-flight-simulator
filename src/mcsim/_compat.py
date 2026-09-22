"""Compatibilità RocketPy 1.1.x ↔ NumPy ≥ 2.4.

RocketPy fino alla 1.1.x chiama ``np.trapz``, rimosso da NumPy 2.4. Invece del
monkey patch silenzioso a livello di modulo del codice originale, lo shim:

- viene applicato una sola volta e solo se ``np.trapz`` manca davvero;
- punta a ``np.trapezoid``, che ha firma e semantica identiche;
- è registrato nei metadati del run (``numpy_trapz_shim``), così un risultato è
  sempre tracciabile all'ambiente che l'ha prodotto.

La soluzione definitiva è aggiornare RocketPy all'ultima 1.x: vedi README.
"""

from __future__ import annotations

import numpy as np

SHIM_APPLIED: bool = False

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):  # pragma: no cover - dipende dall'ambiente
    np.trapz = np.trapezoid  # type: ignore[attr-defined]  # noqa: NPY201 - shim voluto
    SHIM_APPLIED = True
