from pathlib import Path

import pytest

from mcsim.config import DATA_DIR, SimConfig, load_config

DEFAULT_TOML = Path(__file__).resolve().parents[1] / "config" / "default.toml"


def test_default_toml_matches_code_defaults() -> None:
    """Il TOML di riferimento e i default del codice non devono divergere."""
    cfg = load_config(DEFAULT_TOML)
    default = SimConfig()
    assert cfg.site == default.site
    assert cfg.launch == default.launch
    assert cfg.motor == default.motor
    assert cfg.rocket == default.rocket
    assert {k: v.to_dict() for k, v in cfg.stochastic.items()} == {
        k: v.to_dict() for k, v in default.stochastic.items()
    }


def test_relative_paths_resolve_next_to_the_toml() -> None:
    cfg = load_config(DEFAULT_TOML)
    assert cfg.motor.thrust_curve_csv == DATA_DIR / "thrust_curve_M1670.csv"
    assert cfg.rocket.power_off_drag_csv.is_file()


def test_data_files_exist() -> None:
    SimConfig().check_data_files()


def test_missing_data_file_is_reported(tmp_path: Path) -> None:
    f = tmp_path / "broken.toml"
    f.write_text('[motor]\nthrust_curve_csv = "assente.csv"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"assente\.csv"):
        load_config(f).check_data_files()


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[launch]\nrail_lenght_m = 6.0\n", encoding="utf-8")
    with pytest.raises(KeyError, match="rail_lenght_m"):
        load_config(bad)


def test_components_can_be_removed(tmp_path: Path) -> None:
    f = tmp_path / "minimal.toml"
    f.write_text("[rocket]\ndrogue = false\ntail = false\nrail_buttons = false\n[stochastic]\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg.rocket.drogue is None and cfg.rocket.tail is None and cfg.rocket.rail_buttons is None
    assert cfg.stochastic == {}
