"""La campagna di esempio versionata deve restare leggibile e coerente.

È quella che la dashboard mostra al primo avvio (anche su Streamlit Community
Cloud): se il formato di ``results.csv`` cambia senza rigenerarla, il test fallisce
prima che lo scopra chi apre l'app.
"""

from pathlib import Path

from mcsim.stats import summarize_run
from mcsim.storage import list_runs, load_run

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_run_is_discoverable() -> None:
    runs = list_runs(EXAMPLES)
    assert len(runs) == 1, "atteso un solo run di esempio nel repository"


def test_example_run_summarizes() -> None:
    df, metadata = load_run(list_runs(EXAMPLES)[0])
    summary = summarize_run(df)
    assert summary.n_total == metadata["n_flights"] == 200
    assert summary.success_rate == 1.0
    assert 2500.0 < summary.apogee.mean < 3600.0
    assert summary.landing_nominal is not None
    assert summary.landing_nominal.ellipse is not None


def test_example_metadata_has_relative_data_paths() -> None:
    """I percorsi nei metadati non devono puntare alla macchina che ha generato il run."""
    _, metadata = load_run(list_runs(EXAMPLES)[0])
    paths = [
        metadata["config"]["motor"]["thrust_curve_csv"],
        metadata["config"]["rocket"]["power_off_drag_csv"],
        metadata["config"]["rocket"]["power_on_drag_csv"],
    ]
    assert all(p.startswith("data/") for p in paths)
