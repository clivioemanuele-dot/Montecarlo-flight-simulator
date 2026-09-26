"""Dashboard Streamlit del simulatore Monte Carlo 6-DOF.

La pagina legge le campagne salvate su disco da ``python -m mcsim`` e ne mostra
statistiche, grafici e dati grezzi. Se non è ancora stata eseguita nessuna
campagna, mostra quella di esempio inclusa nel progetto (cartella ``examples/``).

Avvio::

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_RUNS = PROJECT_ROOT / "examples"

# Il codice del repository ha la precedenza su un'eventuale copia installata del
# pacchetto: su Streamlit Community Cloud un push aggiorna i file ma non reinstalla
# le dipendenze, quindi senza questa riga la dashboard userebbe la versione vecchia.
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcsim.config import SimConfig  # noqa: E402
from mcsim.plotting import DARK, plot_convergence, plot_dispersion, plot_histogram  # noqa: E402
from mcsim.runner import run_monte_carlo  # noqa: E402
from mcsim.stats import dispersion_ellipse, summarize_run  # noqa: E402
from mcsim.status import FlightStatus  # noqa: E402
from mcsim.storage import list_runs, load_run  # noqa: E402

st.set_page_config(page_title="Monte Carlo 6-DOF — dispersione", page_icon="🚀", layout="wide")

STATUS_LABELS = {
    FlightStatus.OK.value: "Volo completo con apertura del paracadute principale",
    FlightStatus.BALLISTIC.value: "Impatto balistico (paracadute principale non aperto)",
    FlightStatus.NO_RAIL_EXIT.value: "Razzo non partito dalla rampa",
    FlightStatus.NOT_LANDED.value: "Nessun impatto entro il tempo massimo simulato",
    FlightStatus.NON_FINITE.value: "Risultati non validi (divergenza numerica)",
    FlightStatus.ERROR.value: "Errore durante la simulazione",
}

CONFIDENCE_LEVELS = {"90%": 0.90, "95% (consigliato)": 0.95, "99%": 0.99}

ELLIPSE_METHODS = {
    "Predizione (tiene conto del numero di voli)": "prediction",
    "Teorica χ² (valida per N grande)": "chi2",
}

GLOSSARY = [
    (
        "Metodo Monte Carlo",
        "Si simulano molti voli variando a caso i parametri incerti, per ottenere non un solo risultato ma la "
        "distribuzione dei risultati possibili.",
    ),
    (
        "Campagna",
        "Un insieme di voli simulati con la stessa configurazione, salvato in una cartella con dati, grafici e report.",
    ),
    (
        "Seed",
        "Il numero da cui parte la generazione dei valori casuali. Ripetendo una campagna con lo stesso seed "
        "si ottengono risultati identici: è ciò che rende l'analisi verificabile.",
    ),
    (
        "Apogeo (AGL)",
        "Quota massima raggiunta, misurata dal suolo del campo di lancio e non dal livello del mare.",
    ),
    (
        "Numero di Mach",
        "Velocità del razzo rapportata alla velocità del suono. Mach 1 è la velocità del suono, circa 340 m/s "
        "a livello del mare.",
    ),
    (
        "Margine statico",
        "Distanza tra centro di pressione e centro di massa, espressa in diametri del razzo (calibri). Tra "
        "1,5 e 3 il razzo è stabile senza essere eccessivamente sensibile al vento.",
    ),
    (
        "Drogue e main",
        "I due paracadute: il drogue si apre all'apogeo e rallenta la discesa, il main si apre più in basso e "
        "porta il razzo a terra a velocità ridotta.",
    ),
    (
        "Impatto balistico",
        "Volo in cui il paracadute principale non si apre: il razzo tocca terra a velocità elevata. È l'esito "
        "più critico per la sicurezza, quindi resta nelle statistiche.",
    ),
    (
        "CEP (raggio 50%)",
        "Raggio attorno al punto medio di caduta che contiene metà degli atterraggi.",
    ),
    (
        "R95 (raggio 95%)",
        "Raggio che contiene il 95% degli atterraggi: è la misura usata per dimensionare l'area di sicurezza.",
    ),
    (
        "Ellisse di predizione",
        "Regione entro cui cadrà un nuovo volo con la probabilità scelta. È più larga dell'ellisse teorica "
        "perché tiene conto del fatto che media e dispersione sono stimate da un numero finito di "
        "simulazioni.",
    ),
    (
        "Intervallo di confidenza",
        "Fascia di valori entro cui si trova il valore vero di una grandezza stimata, con la probabilità indicata.",
    ),
    (
        "Convergenza",
        "Il progressivo stabilizzarsi di una stima all'aumentare del numero di voli simulati.",
    ),
]

SPACE_CSS = """
<style>
.stApp {
    background:
        radial-gradient(1100px 620px at 18% -12%, rgba(57, 135, 229, 0.20), transparent 62%),
        radial-gradient(900px 520px at 88% 4%, rgba(217, 89, 38, 0.12), transparent 58%),
        #070b14;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    opacity: 0.55;
    background-image:
        radial-gradient(1px 1px at 24px 36px, rgba(255,255,255,0.75), transparent),
        radial-gradient(1px 1px at 148px 92px, rgba(255,255,255,0.50), transparent),
        radial-gradient(1.6px 1.6px at 232px 168px, rgba(255,255,255,0.38), transparent),
        radial-gradient(1px 1px at 318px 64px, rgba(255,255,255,0.45), transparent),
        radial-gradient(1.2px 1.2px at 78px 244px, rgba(255,255,255,0.32), transparent);
    background-size: 380px 380px;
}
[data-testid="stAppViewContainer"], section[data-testid="stSidebar"] { position: relative; z-index: 1; }
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 22, 0.94);
    border-right: 1px solid rgba(92, 132, 200, 0.22);
}
h1 { letter-spacing: -0.02em; }
h2, h3 { letter-spacing: -0.01em; }
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(23, 34, 58, 0.88), rgba(12, 17, 30, 0.88));
    border: 1px solid rgba(92, 132, 200, 0.28);
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.38);
}
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid rgba(92, 132, 200, 0.25);
}
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; }
.mission-badges { margin: 0 0 14px 0; }
.mission-badges span {
    display: inline-block;
    margin: 0 8px 8px 0;
    padding: 4px 11px;
    border-radius: 999px;
    border: 1px solid rgba(92, 132, 200, 0.35);
    background: rgba(23, 34, 58, 0.6);
    font-size: 0.80rem;
    letter-spacing: 0.02em;
    color: #cbd8ef;
}
.tab-intro {
    border-left: 3px solid #3987e5;
    background: rgba(23, 34, 58, 0.45);
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin-bottom: 16px;
    color: #c9d5ea;
    font-size: 0.92rem;
}
</style>
"""

st.markdown(SPACE_CSS, unsafe_allow_html=True)


def intro(text: str) -> None:
    """Riquadro che spiega, in cima a ogni scheda, che cosa si sta guardando."""
    st.markdown(f'<div class="tab-intro">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- caricamento
@st.cache_data(show_spinner=False)
def load_campaign(run_dir: str, mtime: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carica dati e metadati di una campagna. ``mtime`` serve a invalidare la cache."""
    return load_run(run_dir)


def describe_run(run_dir: Path) -> str:
    """Etichetta leggibile per il menu di selezione."""
    try:
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        started = datetime.fromisoformat(meta["started_utc"]).strftime("%d/%m/%Y %H:%M")
        return f"{started} · {meta['n_flights']} voli · seed {meta['seed']}"
    except (OSError, KeyError, ValueError):
        return run_dir.name


def duration_hint(n_flights: int, workers: int) -> str:
    seconds = 3.0 * n_flights / max(workers, 1)
    return f"circa {seconds / 60:.0f} minuti" if seconds >= 90 else f"circa {seconds:.0f} secondi"


# --------------------------------------------------------------------------- barra laterale
st.sidebar.title("🚀 Controlli")

st.sidebar.header("Campagna da analizzare")
with st.sidebar.expander("Cartella dei risultati", expanded=False):
    folder = st.text_input(
        "Percorso",
        value="runs",
        help="Cartella in cui il simulatore salva le campagne. Se è relativa, parte dalla cartella del progetto.",
    )
runs_dir = Path(folder) if Path(folder).is_absolute() else PROJECT_ROOT / folder

runs = list_runs(runs_dir)
showing_example = False
if not runs:
    runs = list_runs(EXAMPLE_RUNS)
    showing_example = bool(runs)

if not runs:
    st.title("Analisi di dispersione Monte Carlo 6-DOF")
    st.warning(
        f"Nessuna campagna trovata in `{runs_dir}`. Avviane una dalla barra laterale, "
        "oppure da terminale con `python -m mcsim -n 1000 --seed 42`."
    )
    st.stop()

options = [str(p) for p in runs]
labels = {str(p): describe_run(p) for p in runs}
stored = st.session_state.get("selected_run")
index = options.index(stored) if stored in options else 0
selected = st.sidebar.selectbox(
    "Campagna",
    options,
    index=index,
    format_func=lambda p: labels[p],
    help="Ogni voce riporta data, numero di voli simulati e seed della campagna.",
)
st.session_state["selected_run"] = selected

st.sidebar.header("Opzioni di analisi")
level_label = st.sidebar.select_slider(
    "Livello di confidenza",
    options=list(CONFIDENCE_LEVELS),
    value="95% (consigliato)",
    help="Si applica agli intervalli di confidenza delle stime e all'ellisse disegnata sulla mappa.",
)
level = CONFIDENCE_LEVELS[level_label]

ellipse_label = st.sidebar.radio(
    "Metodo dell'ellisse",
    list(ELLIPSE_METHODS),
    help=(
        "L'ellisse di predizione è più larga perché considera che media e dispersione "
        "sono stimate da un numero finito di voli. Con molte simulazioni le due coincidono."
    ),
)
ellipse_method = ELLIPSE_METHODS[ellipse_label]

visible_statuses = st.sidebar.multiselect(
    "Esiti da mostrare nella mappa",
    options=[FlightStatus.OK.value, FlightStatus.BALLISTIC.value],
    default=[FlightStatus.OK.value, FlightStatus.BALLISTIC.value],
    format_func=lambda s: STATUS_LABELS[s],
    help="Utile per isolare gli impatti balistici quando sono presenti.",
)
show_ellipse = st.sidebar.checkbox("Mostra l'area di atterraggio stimata", value=True)

st.sidebar.header("Esegui una nuova campagna")
with st.sidebar.form("nuova_campagna"):
    n_flights = st.number_input(
        "Numero di voli", min_value=20, max_value=500, value=50, step=10, help="Più voli, stime più precise."
    )
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    if max_workers > 1:
        workers = st.slider(
            "Processi in parallelo",
            min_value=1,
            max_value=max_workers,
            value=min(2, max_workers),
            help="Più processi riducono i tempi. Conviene lasciare un processore libero.",
        )
    else:
        workers = 1
        st.caption("Un solo processore disponibile: la simulazione viene eseguita in serie.")
    launched = st.form_submit_button("Avvia simulazione", width="stretch")
    st.caption(
        f"Durata stimata: {duration_hint(int(n_flights), int(workers))}. Il seed viene generato "
        "automaticamente e registrato nei metadati, così la campagna resta riproducibile."
    )

if launched:
    auto_seed = int(datetime.now(UTC).timestamp())
    with st.status(f"Simulazione di {int(n_flights)} voli in corso…", expanded=False) as box:
        output = run_monte_carlo(SimConfig(), int(n_flights), auto_seed, runs_dir, int(workers), progress=False)
        box.update(label=f"Campagna completata: {output.run_dir.name}", state="complete")
    st.session_state["selected_run"] = str(output.run_dir)
    st.session_state["last_run_note"] = (
        f"Campagna di {int(n_flights)} voli completata. Seed assegnato: {auto_seed}, "
        "registrato nei metadati per poterla ripetere identica."
    )
    st.rerun()

# --------------------------------------------------------------------------- dati
run_path = Path(selected)
df, meta = load_campaign(selected, (run_path / "results.csv").stat().st_mtime)
summary = summarize_run(df, level)
landing = summary.landing_nominal
ok_flights = df[df["status"] == FlightStatus.OK.value]
physical = df[df["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])]

st.title("🚀 Analisi di dispersione Monte Carlo 6-DOF")
st.markdown(
    '<div class="mission-badges">'
    f"<span>Campagna: {labels[selected]}</span>"
    f"<span>RocketPy {meta.get('versions', {}).get('rocketpy', 'n/d')}</span>"
    f"<span>Durata del calcolo: {meta.get('wall_time_s', 0):,.0f} s</span>"
    "<span>Dati di simulazione, non misure di volo reali</span>"
    "</div>",
    unsafe_allow_html=True,
)

note = st.session_state.pop("last_run_note", None)
if note:
    st.success(note)
if showing_example:
    st.info(
        "Nessuna campagna eseguita finora: quella mostrata è la campagna di esempio inclusa nel progetto. "
        "Avviane una dalla barra laterale per usare dati tuoi."
    )

with st.expander("Come leggere questa dashboard", expanded=False):
    st.markdown(
        """
Il simulatore ripete lo stesso lancio molte volte, cambiando a ogni ripetizione i parametri che nella
realtà non si conoscono con esattezza: vento, massa, spinta del motore, resistenza aerodinamica,
comportamento dei paracadute. Il risultato non è un singolo volo, ma la distribuzione dei voli possibili.

**Le schede**

- **Panoramica** — i numeri principali della campagna: quanto sale il razzo, quanto è stabile, come sono finiti i voli.
- **Mappa di atterraggio** — dove cade il razzo, volo per volo, con l'area che contiene
  la percentuale scelta di atterraggi.
- **Distribuzioni** — come si distribuisce una singola grandezza sui voli simulati.
- **Convergenza** — se il numero di voli simulati è sufficiente perché le stime siano stabili.
- **Configurazione** — razzo, motore e incertezze usati per generare la campagna.
- **Dati** — la tabella completa, volo per volo, con i file da scaricare.
- **Glossario** — il significato dei termini tecnici usati nella pagina.

**I controlli a sinistra**

Scegli la campagna da analizzare, regola il livello di confidenza e il metodo dell'ellisse, decidi quali
esiti mostrare sulla mappa. Dalla stessa barra puoi lanciare una nuova campagna: indichi quanti voli
simulare e quanti processi usare, il resto è automatico.
        """
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Voli simulati", f"{summary.n_total}", help="Numero di lanci simulati in questa campagna.")
col2.metric(
    "Voli riusciti",
    f"{summary.success_rate:.1%}",
    help="Percentuale di voli completati con apertura regolare del paracadute principale.",
)
col3.metric(
    "Apogeo medio",
    f"{summary.apogee.mean:,.0f} m",
    help=f"Quota massima sul suolo. Intervallo al {level:.0%}: "
    f"{summary.apogee.mean_ci[0]:,.0f}–{summary.apogee.mean_ci[1]:,.0f} m.",
)
col4.metric(
    "Raggio 95% atterraggi",
    f"{landing.r95_m:,.0f} m" if landing else "—",
    help="Raggio attorno al punto medio di caduta che contiene il 95% degli atterraggi simulati.",
)

panoramica, mappa, distribuzioni, convergenza, configurazione, dati, glossario = st.tabs(
    ["Panoramica", "Mappa di atterraggio", "Distribuzioni", "Convergenza", "Configurazione", "Dati", "Glossario"]
)

# --------------------------------------------------------------------------- panoramica
with panoramica:
    intro(
        "Le prestazioni medie del razzo su tutti i voli simulati, con l'incertezza di ciascuna stima, "
        "e il riepilogo di come sono finiti i voli."
    )
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Prestazioni di volo")
        rows = [
            ("Apogeo sul suolo", summary.apogee, "m", 0),
            ("Numero di Mach massimo", summary.max_mach, "", 2),
            ("Margine statico al decollo", summary.static_margin, "calibri", 2),
            ("Velocità all'apertura del paracadute principale", summary.main_deploy_speed, "m/s", 1),
        ]
        table = pd.DataFrame(
            [
                {
                    "Grandezza": name,
                    "Media": f"{s.mean:,.{d}f} {unit}".strip(),
                    f"Intervallo al {level:.0%}": f"{s.mean_ci[0]:,.{d}f} – {s.mean_ci[1]:,.{d}f}",
                    "Minimo": f"{s.minimum:,.{d}f}",
                    "Massimo": f"{s.maximum:,.{d}f}",
                }
                for name, s, unit, d in rows
                if s.n > 1
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "L'intervallo indica dove si trova il valore medio vero: più voli si simulano, più è stretto. "
            "Minimo e massimo sono invece i casi estremi incontrati."
        )
    with right:
        st.subheader("Esito dei voli")
        counts = pd.DataFrame(
            [
                {"Esito": STATUS_LABELS[status], "Voli": count, "Quota": f"{count / summary.n_total:.1%}"}
                for status, count in summary.status_counts.items()
                if count
            ]
        )
        st.dataframe(counts, hide_index=True, width="stretch")
        st.caption(
            "I voli non validi sono esclusi dalle statistiche; gli impatti balistici, che sono "
            "fisicamente possibili, restano nel campione perché rappresentano il caso peggiore."
        )

# --------------------------------------------------------------------------- mappa
with mappa:
    intro(
        "Ogni punto è il luogo in cui è atterrato un volo simulato. La rampa di lancio è nell'origine: "
        "verso destra si va a est, verso l'alto a nord. L'ellisse tratteggiata è l'area che dovrebbe "
        "contenere la percentuale di atterraggi scelta nella barra laterale."
    )
    if landing is None or ok_flights.empty:
        st.warning("Voli riusciti insufficienti per stimare l'area di atterraggio.")
    else:
        ellipse = None
        if show_ellipse:
            try:
                ellipse = dispersion_ellipse(
                    ok_flights["impact_x_m"].to_numpy(float),
                    ok_flights["impact_y_m"].to_numpy(float),
                    level,
                    ellipse_method,  # type: ignore[arg-type]
                )
            except ValueError as exc:
                st.warning(f"Ellisse non calcolabile: {exc}")

        plot_df = df[df["status"].isin(visible_statuses)] if visible_statuses else df.iloc[0:0]
        st.pyplot(plot_dispersion(plot_df, ellipse, palette=DARK), clear_figure=True)

        if ellipse is not None and not ellipse.gaussian_consistent:
            st.warning(
                f"L'ellisse contiene il {ellipse.empirical_coverage:.1%} dei punti invece del "
                f"{level:.0%} previsto: la distribuzione degli atterraggi non è gaussiana. "
                "Per l'area di sicurezza conviene usare il raggio empirico riportato qui sotto."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Raggio 50% (CEP)", f"{landing.cep50_m:,.0f} m", help="Metà degli atterraggi cade entro questo raggio."
        )
        m2.metric("Raggio 95%", f"{landing.r95_m:,.0f} m", help="Contiene il 95% degli atterraggi.")
        m3.metric(
            "Distanza dalla rampa (95%)",
            f"{landing.range_from_pad.p95:,.0f} m",
            help="Il 95% dei voli atterra entro questa distanza dal punto di lancio.",
        )
        m4.metric(
            "Distanza massima",
            f"{summary.max_range_all_physical_m:,.0f} m",
            help="Atterraggio più lontano osservato nella campagna.",
        )

        if ellipse is not None:
            st.caption(
                f"Ellisse {ellipse_label.lower()}: semiassi {ellipse.semi_major_m:,.0f} m e "
                f"{ellipse.semi_minor_m:,.0f} m, orientata di {ellipse.angle_deg:.0f}° rispetto all'est, "
                f"copertura effettiva {ellipse.empirical_coverage:.1%}."
            )

# --------------------------------------------------------------------------- distribuzioni
with distribuzioni:
    intro(
        "Scegli una grandezza e osserva come si distribuisce sui voli simulati: la linea chiara segna "
        "il valore medio, quella arancione il 95° percentile, cioè il valore superato solo dal 5% dei voli."
    )
    metrics: dict[str, tuple[pd.Series, str]] = {
        "Apogeo sul suolo": (physical["apogee_agl_m"], "m"),
        "Distanza dalla rampa": (
            pd.Series(np.hypot(physical["impact_x_m"], physical["impact_y_m"]), index=physical.index),
            "m",
        ),
        "Durata del volo": (physical["flight_time_s"], "s"),
        "Numero di Mach massimo": (physical["max_mach"], ""),
        "Velocità di uscita dalla rampa": (physical["rail_exit_speed_ms"], "m/s"),
        "Velocità all'apertura del paracadute": (ok_flights["main_deploy_speed_ms"], "m/s"),
        "Velocità di impatto al suolo": (physical["impact_speed_ms"], "m/s"),
        "Margine statico al decollo": (physical["static_margin_liftoff_cal"], "calibri"),
    }
    choice_col, bins_col = st.columns([3, 2])
    choice = choice_col.selectbox("Grandezza da visualizzare", list(metrics))
    bins = bins_col.slider(
        "Dettaglio dell'istogramma",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
        help="Numero di barre: più barre mostrano più dettaglio, meno barre una forma più leggibile.",
    )
    values, unit = metrics[choice]
    st.pyplot(plot_histogram(values, choice, unit, bins, palette=DARK), clear_figure=True)

    clean = values[np.isfinite(values)]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Voli": int(clean.size),
                    "Media": f"{clean.mean():,.2f}",
                    "Deviazione standard": f"{clean.std(ddof=1):,.2f}",
                    "5° percentile": f"{clean.quantile(0.05):,.2f}",
                    "Mediana": f"{clean.median():,.2f}",
                    "95° percentile": f"{clean.quantile(0.95):,.2f}",
                    "Minimo": f"{clean.min():,.2f}",
                    "Massimo": f"{clean.max():,.2f}",
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "La deviazione standard misura quanto i valori si allontanano in media dal valore centrale: "
        "più è piccola, più il comportamento del razzo è ripetibile."
    )

# --------------------------------------------------------------------------- convergenza
with convergenza:
    intro(
        "Questo grafico risponde a una domanda pratica: quanti voli servono? La curva mostra come cambia "
        "la stima dell'apogeo medio man mano che si aggiungono simulazioni."
    )
    if len(physical) > 2:
        st.pyplot(plot_convergence(physical["apogee_agl_m"], palette=DARK), clear_figure=True)
        st.caption(
            "Quando la curva si appiattisce e la fascia di incertezza smette di restringersi in modo "
            "apprezzabile, simulare altri voli non aggiunge informazione utile."
        )
    else:
        st.info("Servono almeno tre voli validi per il grafico di convergenza.")

# --------------------------------------------------------------------------- configurazione
with configurazione:
    intro(
        "Il modello usato per generare questa campagna: come è stata eseguita, com'è fatto il razzo e "
        "quali parametri sono stati trattati come incerti."
    )
    config = meta.get("config", {})
    st.subheader("Esecuzione")
    st.dataframe(
        pd.DataFrame(
            [
                {"Voce": "Seed", "Valore": str(meta.get("seed", "—"))},
                {"Voce": "Voli simulati", "Valore": str(meta.get("n_flights", "—"))},
                {"Voce": "Processi in parallelo", "Valore": str(meta.get("workers", "—"))},
                {"Voce": "Durata del calcolo", "Valore": f"{meta.get('wall_time_s', float('nan')):,.0f} s"},
                {"Voce": "Avvio (UTC)", "Valore": str(meta.get("started_utc", "—"))},
                {"Voce": "RocketPy", "Valore": str(meta.get("versions", {}).get("rocketpy", "—"))},
                {"Voce": "Python", "Valore": str(meta.get("versions", {}).get("python", "—"))},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Modello del razzo")
    rocket, motor, launch = config.get("rocket", {}), config.get("motor", {}), config.get("launch", {})
    st.dataframe(
        pd.DataFrame(
            [
                {"Parametro": "Massa a secco senza motore", "Valore": f"{rocket.get('mass_kg', '—')} kg"},
                {"Parametro": "Diametro del corpo", "Valore": f"{2 * rocket.get('radius_m', 0):.3f} m"},
                {"Parametro": "Lunghezza dell'ogiva", "Valore": f"{rocket.get('nose_length_m', '—')} m"},
                {"Parametro": "Numero di alette", "Valore": str(rocket.get("fin_n", "—"))},
                {"Parametro": "Massa a secco del motore", "Valore": f"{motor.get('dry_mass_kg', '—')} kg"},
                {"Parametro": "Tempo di combustione", "Valore": f"{motor.get('burn_time_s', '—')} s"},
                {"Parametro": "Lunghezza della rampa", "Valore": f"{launch.get('rail_length_m', '—')} m"},
                {"Parametro": "Inclinazione della rampa", "Valore": f"{launch.get('inclination_deg', '—')}°"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Parametri incerti")
    st.caption(
        "Sono le grandezze fatte variare a ogni volo. La distribuzione indica come vengono estratti i valori: "
        "troncata per le grandezze con limiti fisici, lognormale per quelle sempre positive, "
        "di Weibull e von Mises per intensità e direzione del vento."
    )
    stochastic = config.get("stochastic", {})
    if stochastic:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Parametro": name,
                        "Distribuzione": str(spec.get("kind", "")),
                        "Valori": ", ".join(f"{k} = {v}" for k, v in spec.items() if k != "kind"),
                    }
                    for name, spec in stochastic.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Configurazione completa (JSON)"):
        st.json(config)

# --------------------------------------------------------------------------- dati
with dati:
    intro(
        "La tabella completa, una riga per volo simulato. Le colonne che iniziano con `in_` sono i valori "
        "estratti per quel volo, le altre sono i risultati ottenuti."
    )
    only_failed = st.checkbox("Mostra solo i voli non riusciti", value=False)
    table = df[~df["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])] if only_failed else df
    st.dataframe(table, width="stretch", height=420)

    files = st.columns(3)
    files[0].download_button(
        "Scarica i risultati (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
        width="stretch",
    )
    files[1].download_button(
        "Scarica i metadati (JSON)",
        json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="metadata.json",
        mime="application/json",
        width="stretch",
    )
    report = run_path / "report.tex"
    if report.is_file():
        files[2].download_button(
            "Scarica il report (LaTeX)",
            report.read_bytes(),
            file_name="report.tex",
            mime="text/plain",
            width="stretch",
        )

# --------------------------------------------------------------------------- glossario
with glossario:
    intro("Il significato dei termini usati nella pagina, senza presupporre conoscenze di dinamica del volo.")
    st.dataframe(
        pd.DataFrame([{"Termine": term, "Significato": meaning} for term, meaning in GLOSSARY]),
        hide_index=True,
        width="stretch",
        height=520,
    )
