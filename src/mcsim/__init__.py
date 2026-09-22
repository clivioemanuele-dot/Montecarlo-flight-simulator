"""mcsim — analisi di dispersione Monte Carlo 6-DOF per razzi sperimentali (RocketPy).

Moduli:

- :mod:`mcsim.config`        configurazione tipizzata (modello nominale, incertezze, lancio)
- :mod:`mcsim.distributions` distribuzioni di probabilità dei parametri incerti
- :mod:`mcsim.model`         factory pure Environment / SolidMotor / Rocket / Flight
- :mod:`mcsim.sampling`      campionamento riproducibile (SeedSequence per volo e parametro)
- :mod:`mcsim.runner`        esecuzione parallela, isolamento errori, classificazione esiti
- :mod:`mcsim.stats`         statistiche con intervalli di confidenza, ellissi, R95/CEP
- :mod:`mcsim.plotting`      grafici (Figure senza stato globale pyplot)
- :mod:`mcsim.report`        report LaTeX con escaping e gestione NaN
- :mod:`mcsim.storage`       cartelle di run, scritture atomiche, checkpoint, caricamento
"""

__version__ = "0.2.0"
