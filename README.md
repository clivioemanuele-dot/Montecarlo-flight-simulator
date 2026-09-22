# poli_mc_sim — Simulatore di dispersione Monte Carlo 6-DOF

Analisi di dispersione per razzi sperimentali basata su [RocketPy](https://docs.rocketpy.org):
campionamento stocastico riproducibile, esecuzione parallela con isolamento degli errori,
statistica con intervalli di confidenza, report LaTeX e dashboard.

## Modello di riferimento

I parametri di default descrivono un vettore realmente volato e documentato, non valori inventati.

| Elemento | Valore | Origine |
| --- | --- | --- |
| Vettore | *Calisto* — ogiva von Kármán 0,558 m, 4 alette trapezoidali con cant 0,5°, boattail, rail buttons, massa a secco 14,426 kg | Razzo sperimentale del team universitario Projeto Jupiter, caso di riferimento della documentazione RocketPy |
| Motore | Cesaroni Pro75 **M1670 Blue Streak**: 6026,35 N·s, 1545,22 N medi, 2200 N di picco, 3,9 s, propellente 2,956 kg su 5,231 kg totali (Isp 208 s) | Dati di certificazione pubblici (ThrustCurve.org) |
| Aerodinamica | Curve Cd(Mach) power-off e power-on, 0,01 ≤ M ≤ 1,5 | Calcolate da `mcsim.aero` sulla geometria reale: attrito Prandtl–Schlichting, base drag di Hoerner, salita transonica, margine protuberanze |
| Recupero | Drogue all'apogeo (CdS 1,0 m²) e main a 800 m AGL (CdS 10,0 m²), ritardo 1,5 s | Configurazione dual-deploy del vettore |

Volo nominale risultante (rampa 5,2 m a 85°, vento nullo): **apogeo 3159 m AGL**, Mach massimo 0,82,
margine statico 2,06 cal al decollo, uscita dalla rampa a 28 m/s, apertura del main a 16 m/s,
impatto a 4,9 m/s dopo 323 s.

## Installazione

```powershell
cd poli_mc_sim
python -m venv venv
venv\Scripts\activate
pip install -e ".[app,dev]"
```

Funziona con RocketPy 1.1.3 (con uno shim esplicito per `np.trapz`, registrato nei metadati del run)
e con RocketPy 1.13. **Consigliato `pip install -U rocketpy`**: stessi risultati e circa 3,4 volte più veloce.

## Uso

```powershell
python -m mcsim -n 1000 --seed 42                          # campagna completa, parallela (CPU-1 processi)
python -m mcsim -n 200 --seed 42 --config config\default.toml
python -m mcsim.aero                                        # rigenera le curve Cd(Mach) in data\
streamlit run app.py                                        # dashboard sui run salvati
pytest                                                      # 57 test (6 lenti, integrano voli reali)
pytest -m "not slow"                                        # solo i test veloci
```

Un volo costa ~3,0 s con RocketPy 1.1.3 e ~0,9 s con la 1.13: 1000 voli su 8 core richiedono
circa 7 minuti, o 2 minuti con la versione aggiornata.

Ogni run crea `runs\<data>_seed<seed>_n<N>\` con:

| File | Contenuto |
| --- | --- |
| `results.csv` | un record per volo: input campionati (`in_*`), esito (`status`), metriche, errore |
| `metadata.json` | seed, N, worker, versioni dei pacchetti, configurazione completa, tempi, esiti |
| `dispersion.png`, `convergence.png` | mappa degli impatti con ellisse di predizione; convergenza della media |
| `report.tex` | report LaTeX compilabile (`pdflatex report.tex`) |

Stesso seed e stessa configurazione producono lo stesso `results.csv`, con qualunque numero di worker.

La dashboard non è mai vuota: se `runs\` non contiene ancora nulla mostra la campagna di esempio
in `examples\` (200 voli, seed 20261025), segnalandolo in cima alla pagina.

## Deploy su Streamlit Community Cloud

1. Carica il repository su GitHub (la cartella `runs\` è ignorata: i risultati non finiscono nel repo).
2. Su [share.streamlit.io](https://share.streamlit.io) crea l'app indicando il repository, il branch
   e `app.py` come file principale; scegli Python 3.11 o successivo (serve `tomllib`).
3. `requirements.txt` installa il progetto stesso (riga `.`) e con esso tutte le dipendenze.

Da lì in poi ogni push aggiorna l'app da solo; se cambi `requirements.txt` il Cloud rifà il deploy
completo. Online gira la dashboard, non le campagne grandi: una CPU condivisa e il disco temporaneo
bastano per la campagna di esempio e per una prova da qualche decina di voli, non per 1000.

## Struttura

```
poli_mc_sim/
  app.py                  dashboard Streamlit (legge i run salvati, non genera dati)
  pyproject.toml          pacchetto, dipendenze, extra [app] e [dev], ruff e mypy
  config/default.toml     configurazione di riferimento, uguale ai default del codice
  data/                   curva di spinta e curve Cd(Mach) + README con la provenienza
  src/mcsim/
    aero.py               stima Cd(Mach) per componenti e generazione dei CSV
    config.py             configurazione tipizzata, unità nei nomi, caricamento TOML
    distributions.py      distribuzioni (truncnorm, lognormal, weibull, von Mises...) con validazione
    model.py              factory pure Environment / SolidMotor / Rocket / Flight, profilo di vento
    sampling.py           semi per (volo, parametro) con SeedSequence
    runner.py             esecuzione parallela, isolamento errori, classificazione esiti, checkpoint
    stats.py              IC su media e quantili, Clopper-Pearson, ellisse di predizione, R95/CEP
    plotting.py           grafici senza stato globale pyplot
    report.py             report LaTeX con escaping e gestione NaN
    storage.py            cartelle di run, scritture atomiche, caricamento
    status.py             esiti possibili di un volo
    cli.py                riga di comando
  examples/               campagna di esempio versionata (200 voli), mostrata dalla dashboard
  tests/                  57 test pytest
  runs/                   output delle campagne (ignorato da git)
```

## Modello stocastico

Tredici parametri incerti, ciascuno con la distribuzione adatta al suo supporto fisico
(`config/default.toml`, sezione `[stochastic]`): inclinazione e azimut di rampa, massa a secco,
impulso e tempo di combustione, Cd, CdS e ritardi dei due paracadute, velocità, direzione e
profilo verticale del vento. Nessuna normale non troncata su grandezze positive o limitate.

Gli ordini di grandezza seguono la pratica industriale per un razzo di classe M. **Prima di usare
i risultati per una campagna reale** vanno confermati con i dati del tuo vettore: pesate, lotto e
temperatura del motore, CFD o galleria del vento, prove di apertura del recupero, climatologia o
previsione ensemble del sito di lancio.
