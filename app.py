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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from mcsim.config import SimConfig
from mcsim.plotting import plot_convergence, plot_dispersion, plot_histogram
from mcsim.runner import run_monte_carlo
from mcsim.stats import dispersion_ellipse, summarize_run
from mcsim.status import FlightStatus
from mcsim.storage import list_runs, load_run

st.set_page_config(page_title="Monte Carlo 6-DOF — dispersione", page_icon="🚀", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_RUNS = PROJECT_ROOT / "examples"

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
    return f"circa {seconds / 60:.0f} min" if seconds >= 90 else f"circa {seconds:.0f} s"


# --------------------------------------------------------------------------- barra laterale
st.sidebar.title("Impostazioni")

st.sidebar.header("Campagna da analizzare")
with st.sidebar.expander("Cartella dei risultati", expanded=False):
    folder = st.text_input(
        "Percorso",
        value="runs",
        help="Cartella in cui il simulatore salva le campagne. Relativa alla cartella del progetto.",
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
        f"Nessuna campagna trovata in `{runs_dir}`. Eseguine una dalla barra laterale, "
        "oppure da terminale con `python -m mcsim -n 1000 --seed 42`."
    )
    st.stop()

options = [str(p) for p in runs]
labels = {str(p): describe_run(p) for p in runs}
stored = st.session_state.get("selected_run")
index = options.index(stored) if stored in options else 0
selected = st.sidebar.selectbox("Campagna", options, index=index, format_func=lambda p: labels[p])
st.session_state["selected_run"] = selected

st.sidebar.header("Opzioni di analisi")
level_label = st.sidebar.select_slider(
    "Livello di confidenza",
    options=list(CONFIDENCE_LEVELS),
    value="95% (consigliato)",
    help="Si applica agli intervalli di confidenza e all'ellisse disegnata sulla mappa.",
)
level = CONFIDENCE_LEVELS[level_label]

ellipse_label = st.sidebar.radio(
    "Metodo dell'ellisse",
    list(ELLIPSE_METHODS),
    help=(
        "L'ellisse di predizione è più larga perché considera che media e dispersione "
        "sono stimate da un campione finito. Con molti voli le due coincidono."
    ),
)
ellipse_method = ELLIPSE_METHODS[ellipse_label]

visible_statuses = st.sidebar.multiselect(
    "Esiti da mostrare nella mappa",
    options=[FlightStatus.OK.value, FlightStatus.BALLISTIC.value],
    default=[FlightStatus.OK.value, FlightStatus.BALLISTIC.value],
    format_func=lambda s: STATUS_LABELS[s],
)
show_ellipse = st.sidebar.checkbox("Mostra l'area di atterraggio stimata", value=True)

st.sidebar.header("Esegui una nuova campagna")
with st.sidebar.form("nuova_campagna"):
    n_flights = st.number_input("Numero di voli", min_value=20, max_value=500, value=50, step=10)
    seed = st.number_input(
        "Seed", min_value=0, value=42, step=1, help="Stesso seed e stessa configurazione danno gli stessi risultati."
    )
    max_workers = max(1, (os.cpu_count() or 2) - 1)
    if max_workers > 1:
        workers = st.slider(
            "Processi in parallelo",
            min_value=1,
            max_value=max_workers,
            value=min(2, max_workers),
            help="Più processi riducono i tempi. Lascia almeno un core libero per il resto del sistema.",
        )
    else:
        workers = 1
        st.caption("Un solo processore disponibile: la simulazione viene eseguita in serie.")
    launched = st.form_submit_button("Avvia simulazione", width="stretch")
    st.caption(f"Durata stimata: {duration_hint(int(n_flights), int(workers))} (circa 3 s per volo).")

if launched:
    with st.status(f"Simulazione di {int(n_flights)} voli in corso…", expanded=False) as box:
        output = run_monte_carlo(SimConfig(), int(n_flights), int(seed), runs_dir, int(workers), progress=False)
        box.update(label=f"Campagna completata: {output.run_dir.name}", state="complete")
    st.session_state["selected_run"] = str(output.run_dir)
    st.rerun()

# --------------------------------------------------------------------------- dati
run_path = Path(selected)
df, meta = load_campaign(selected, (run_path / "results.csv").stat().st_mtime)
summary = summarize_run(df, level)
landing = summary.landing_nominal
ok_flights = df[df["status"] == FlightStatus.OK.value]

st.title("Analisi di dispersione Monte Carlo 6-DOF")
st.caption(
    "Ogni valore proviene da una campagna di simulazione salvata su disco ed è riproducibile "
    "a partire dal suo seed. Le campagne si scelgono e si lanciano dalla barra laterale."
)
if showing_example:
    st.info(
        "Nessuna campagna eseguita finora: stai vedendo quella di esempio inclusa nel progetto. "
        "Avviane una dalla barra laterale per usare dati tuoi."
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Voli simulati", f"{summary.n_total}", help=f"Seed {meta.get('seed')} · {labels[selected]}")
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

panoramica, mappa, distribuzioni, convergenza, configurazione, dati = st.tabs(
    ["Panoramica", "Mappa di atterraggio", "Distribuzioni", "Convergenza", "Configurazione", "Dati"]
)

# --------------------------------------------------------------------------- panoramica
with panoramica:
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
            "I voli non validi sono esclusi dalle statistiche; gli impatti balistici, "
            "che sono fisicamente possibili, restano nel campione."
        )

# --------------------------------------------------------------------------- mappa
with mappa:
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
        st.pyplot(plot_dispersion(plot_df, ellipse), clear_figure=True)
        st.caption(
            "Ogni punto è l'atterraggio di un volo simulato, con la rampa nell'origine. "
            "L'asse orizzontale indica la distanza verso est, quello verticale verso nord."
        )

        if ellipse is not None and not ellipse.gaussian_consistent:
            st.warning(
                f"L'ellisse contiene il {ellipse.empirical_coverage:.1%} dei punti invece del "
                f"{level:.0%} previsto: la distribuzione degli atterraggi non è gaussiana. "
                "Per l'area di sicurezza usa il raggio empirico riportato qui sotto."
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
        m4.metric("Distanza massima", f"{summary.max_range_all_physical_m:,.0f} m")

        if ellipse is not None:
            st.caption(
                f"Ellisse {ellipse_label.lower()}: semiassi {ellipse.semi_major_m:,.0f} m e "
                f"{ellipse.semi_minor_m:,.0f} m, orientata di {ellipse.angle_deg:.0f}° rispetto all'est, "
                f"copertura effettiva {ellipse.empirical_coverage:.1%}."
            )

# --------------------------------------------------------------------------- distribuzioni
with distribuzioni:
    physical = df[df["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])]
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
    choice = st.selectbox("Grandezza da visualizzare", list(metrics))
    values, unit = metrics[choice]
    bins = st.slider("Numero di intervalli dell'istogramma", min_value=10, max_value=60, value=30, step=5)
    st.pyplot(plot_histogram(values, choice, unit, bins), clear_figure=True)

    stats = summary.apogee if choice == "Apogeo sul suolo" else None
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
    if stats is not None:
        st.caption(
            f"Intervallo di confidenza al {level:.0%} sulla media: {stats.mean_ci[0]:,.1f} – {stats.mean_ci[1]:,.1f} m."
        )

# --------------------------------------------------------------------------- convergenza
with convergenza:
    physical = df[df["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])]
    if len(physical) > 2:
        st.pyplot(plot_convergence(physical["apogee_agl_m"]), clear_figure=True)
        st.caption(
            "La curva mostra come la stima dell'apogeo medio si stabilizza aggiungendo voli. "
            "Quando la banda di incertezza smette di restringersi in modo apprezzabile, "
            "aumentare il numero di simulazioni non aggiunge informazione."
        )
    else:
        st.info("Servono almeno tre voli validi per il grafico di convergenza.")

# --------------------------------------------------------------------------- configurazione
with configurazione:
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
    st.subheader("Risultati per singolo volo")
    only_failed = st.checkbox("Mostra solo i voli non riusciti", value=False)
    table = df[~df["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])] if only_failed else df
    st.dataframe(table, width="stretch", height=420)
    st.caption(
        "Le colonne che iniziano con `in_` sono i valori estratti per quel volo; le altre sono i risultati. "
        "La colonna `status` indica come è finito il volo."
    )

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
