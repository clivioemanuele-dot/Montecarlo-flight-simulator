import math
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from mcsim.report import fmt, latex_escape, render_report
from mcsim.stats import summarize_run


def test_escape_special_characters() -> None:
    assert latex_escape("run_1 50% & #2 {x} ~ ^") == (r"run\_1 50\% \& \#2 \{x\} \textasciitilde{} \textasciicircum{}")
    assert latex_escape(r"C:\runs") == r"C:\textbackslash{}runs"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None])
def test_non_finite_numbers_render_as_dash(value: float | None) -> None:
    assert fmt(value) == "--"


def test_negative_and_thousands() -> None:
    assert fmt(-1234.567, 1) == r"$-1\,234.6$"
    assert fmt(0.953, 1, pct=True) == r"$95.3\,\%$"


def _fake_results(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "status": ["ok"] * (n - 2) + ["ballistic", "error"],
            "apogee_agl_m": rng.normal(2240, 40, n),
            "max_mach": rng.normal(0.65, 0.01, n),
            "main_deploy_speed_ms": rng.normal(16, 1, n),
            "static_margin_liftoff_cal": rng.normal(2.3, 0.01, n),
            "impact_x_m": rng.normal(800, 400, n),
            "impact_y_m": rng.normal(400, 600, n),
        }
    )


def test_report_compiles_with_hostile_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    metadata = {
        "seed": 42,
        "n_flights": 60,
        "workers": 2,
        "wall_time_s": float("nan"),
        "started_utc": "2026-10-25T12:00:00+00:00",
        "versions": {"rocketpy": "1.1.3_dev#1", "numpy": "2.5", "scipy": "1.18", "python": "3.13"},
        "config": {"stochastic": {"drag_scale": {"kind": "lognormal", "median": 1.0, "sigma": 0.08}}},
    }
    tex = render_report(summarize_run(_fake_results()), metadata, figures={})
    assert "nan" not in tex and "inf" not in tex.replace("\\infty", "")
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex non disponibile")
    (tmp_path / "report.tex").write_text(tex, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
