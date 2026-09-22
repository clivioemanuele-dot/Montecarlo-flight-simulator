"""Dashboard Streamlit: mostra **solo** risultati prodotti dal simulatore.

La versione precedente generava numeri con ``np.random.normal`` e li presentava
come "simulazione stocastica" (D-01). Qui la dashboard legge i run salvati da
``python -m mcsim`` e, se richiesto, lancia una piccola campagna reale.

Si apre sempre con dei dati: se non c'è nessuna campagna in ``runs/`` usa quella
di esempio inclusa nel repository (``examples/``), così funziona anche appena
clonata o pubblicata su Streamlit Community Cloud.

Avvio::

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mcsim.config import SimConfig
from mcsim.plotting import plot_convergence, plot_dispersion
from mcsim.runner import run_monte_carlo
from mcsim.stats import summarize_run
from mcsim.status import FlightStatus
from mcsim.storage import list_runs, load_run

st.set_page_config(page_title="Monte Carlo 6-DOF — dispersione", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_RUNS = PROJECT_ROOT / "examples"
"""Campagna di esempio versionata: la dashboard non è mai vuota al primo avvio."""


@st.cache_data(show_spinner=False)
def _load(run_dir: str, mtime: float) -> tuple[pd.DataFrame, dict]:  # mtime invalida la cache
    return load_run(run_dir)


st.title("Analisi di dispersione Monte Carlo 6-DOF")
st.caption("Dati letti dai run salvati su disco: ogni numero è riproducibile da seed e configurazione del run.")

folder = st.sidebar.text_input("Cartella dei run", value="runs")
runs_dir = Path(folder) if Path(folder).is_absolute() else PROJECT_ROOT / folder

with st.sidebar.form("new_run"):
    st.subheader("Nuova campagna")
    n = st.number_input("Voli", min_value=20, max_value=200, value=50, step=10)
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    submitted = st.form_submit_button("Esegui")
    st.caption("Eseguita in serie, ~3 s per volo. Per campagne grandi usare la CLI, che è parallela.")
if submitted:
    with st.status(f"Simulazione di {n} voli in corso…", expanded=False) as box:
        out = run_monte_carlo(SimConfig(), int(n), int(seed), runs_dir, workers=1, progress=False)
        box.update(label=f"Completato: {out.run_dir.name}", state="complete")
    st.session_state["selected_run"] = str(out.run_dir)

runs = list_runs(runs_dir)
if not runs:
    runs = list_runs(EXAMPLE_RUNS)
    if not runs:
        st.info(f"Nessun run in `{runs_dir}`. Lancialo dalla barra laterale o con `python -m mcsim -n 1000 --seed 42`.")
        st.stop()
    st.info(
        "Nessuna campagna in `runs/`: qui sotto c'è quella di esempio inclusa nel repository. "
        "Per una tua campagna: `python -m mcsim -n 1000 --seed 42`, oppure il modulo qui a lato."
    )

names = [str(p) for p in runs]
default = names.index(st.session_state["selected_run"]) if st.session_state.get("selected_run") in names else 0
selected = st.sidebar.selectbox("Run", names, index=default, format_func=lambda p: Path(p).name)
st.session_state["selected_run"] = selected
df, meta = _load(selected, (Path(selected) / "results.csv").stat().st_mtime)
summary = summarize_run(df)

a, land = summary.apogee, summary.landing_nominal
c1, c2, c3, c4 = st.columns(4)
c1.metric("Voli", f"{summary.n_total}", help=f"Seed {meta.get('seed')} · RocketPy {meta['versions'].get('rocketpy')}")
c2.metric("Successo", f"{summary.success_rate:.1%}", help="Status ok; IC 95 % Clopper–Pearson nel report")
c3.metric("Apogeo AGL medio", f"{a.mean:,.0f} m", help=f"IC 95 %: {a.mean_ci[0]:,.0f}–{a.mean_ci[1]:,.0f} m")
c4.metric("R95 impatti", f"{land.r95_m:,.0f} m" if land else "—", help="Raggio attorno al punto medio con il 95 %")

left, right = st.columns([3, 2])
with left:
    ellipse = land.ellipse if land else None
    st.pyplot(plot_dispersion(df, ellipse), clear_figure=True)
    if ellipse is not None and not ellipse.gaussian_consistent:
        st.warning(
            f"Copertura empirica dell'ellisse {ellipse.empirical_coverage:.1%} contro {ellipse.level:.0%} "
            "nominale: distribuzione non gaussiana, usare R95 per l'area di sicurezza."
        )
with right:
    physical = df[df["status"].isin([FlightStatus.OK.value, FlightStatus.BALLISTIC.value])]
    if len(physical) > 2:
        st.pyplot(plot_convergence(physical["apogee_agl_m"]), clear_figure=True)
    st.subheader("Esiti")
    st.dataframe(pd.Series(summary.status_counts, name="voli"))

errors = df[df["status"] == FlightStatus.ERROR.value]
if len(errors):
    with st.expander(f"{len(errors)} voli con errore"):
        st.dataframe(errors[["index", "error"]])

st.download_button("Scarica results.csv", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")
