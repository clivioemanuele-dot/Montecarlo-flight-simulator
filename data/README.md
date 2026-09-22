# Dati di ingresso del simulatore

| File | Contenuto | Origine |
| --- | --- | --- |
| `thrust_curve_M1670.csv` | Curva di spinta del Cesaroni Pro75 M1670 Blue Streak: `tempo [s], spinta [N]` | Ricostruita dai dati di certificazione pubblici (impulso totale 6026,35 N·s, spinta media 1545,22 N, picco 2200 N, combustione 3,9 s) con un profilo BATES leggermente regressivo. L'integrale e il picco coincidono con i valori certificati. |
| `cd_power_off.csv` | Cd in funzione del Mach a motore spento: `Mach, Cd` | Calcolata da `mcsim.aero` sulla geometria reale del vettore (attrito Prandtl–Schlichting, base drag di Hoerner, salita transonica empirica, margine per protuberanze). |
| `cd_power_on.csv` | Cd in funzione del Mach a motore acceso | Come sopra, senza resistenza di base: il getto riempie la scia. |

## Rigenerare le curve di resistenza

```powershell
python -m mcsim.aero
```

Il test `tests/test_aero.py` verifica che i file su disco coincidano con il modello:
se modifichi la geometria in `config/default.toml` devi rigenerarli.

## Sostituire con dati migliori

Per una campagna di volo reale queste curve vanno rimpiazzate con dati misurati o
di alta fedeltà:

- **Spinta**: file `.eng` ufficiale del motore da ThrustCurve.org, convertito in due
  colonne `tempo, spinta`.
- **Resistenza**: esportazione da RASAero II, OpenRocket o CFD, sempre in due colonne
  `Mach, Cd`.

Basta puntare `thrust_curve_csv`, `power_off_drag_csv` e `power_on_drag_csv` del TOML
ai nuovi file: nessuna modifica al codice.
