"""Unit tests for the Phase-5 experiment harness scripts
(``scripts/experiment/{init_experiment,lego_detect_machine}.py``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP_DIR = _REPO_ROOT / "scripts" / "experiment"


@pytest.fixture(scope="module")
def mod():
    if str(_EXP_DIR) not in sys.path:
        sys.path.insert(0, str(_EXP_DIR))
    return (
        importlib.import_module("init_experiment"),
        importlib.import_module("lego_detect_machine"),
    )


# --- lego_detect_machine ----------------------------------------------------

def test_resolve_machine_matches_macbook(mod):
    _, detect = mod
    prof = detect.resolve_machine("Juliens-MacBook-Pro.local")
    assert prof["name"] == "macbook"
    assert prof["jax_platforms"] == "cpu"  # Metal broken -> CPU


def test_resolve_machine_matches_levante(mod):
    _, detect = mod
    prof = detect.resolve_machine("levante1.levante.dkrz.de")
    assert prof["name"] == "levante-gpu"
    assert prof["scheduler"] == "slurm"


def test_resolve_machine_falls_back_to_default(mod):
    _, detect = mod
    prof = detect.resolve_machine("some-unknown-cluster-node-42")
    assert prof["name"] == "default"


# --- init_experiment helpers ------------------------------------------------

def test_coerce_types(mod):
    init, _ = mod
    assert init._coerce("true") is True and init._coerce("False") is False
    assert init._coerce("96") == 96 and isinstance(init._coerce("96"), int)
    assert init._coerce("1e6") == 1e6
    assert init._coerce("none") is None
    assert init._coerce("kpp") == "kpp"


def test_parse_overrides_requires_equals(mod):
    init, _ = mod
    assert init._parse_overrides(["grid.resolution=96"]) == [("grid.resolution", 96)]
    with pytest.raises(SystemExit):
        init._parse_overrides(["grid.resolution"])


def test_template_path_unknown_raises(mod):
    init, _ = mod
    with pytest.raises(SystemExit):
        init._template_path("nope/does_not_exist")


# --- init_experiment end-to-end --------------------------------------------

def test_init_writes_runnable_dir(mod, tmp_path):
    init, _ = mod
    out = tmp_path / "w2"
    rc = init.main(["2d/williamson2_sw", "--name", "w2",
                    "--output-dir", str(out), "--machine", "default"])
    assert rc == 0
    assert (out / "config.yaml").exists()
    assert (out / "run.sh").exists()
    assert (out / "run.yaml").exists()
    run_sh = (out / "run.sh").read_text()
    assert "legoesm run config.yaml" in run_sh
    assert "JAX_PLATFORMS=cpu" in run_sh


def test_init_applies_overrides(mod, tmp_path):
    init, _ = mod
    from legoesm.config import Config
    out = tmp_path / "w2hi"
    init.main(["2d/williamson2_sw", "--name", "w2hi", "--output-dir", str(out),
               "--machine", "default", "-o", "grid.resolution=96"])
    cfg = Config.from_yaml(str(out / "config.yaml"))
    assert cfg.get("grid.resolution") == 96


def test_init_bad_override_fails_fast_no_dir(mod, tmp_path):
    init, _ = mod
    out = tmp_path / "bad"
    with pytest.raises(SystemExit):
        init.main(["2d/williamson2_sw", "--name", "bad", "--output-dir", str(out),
                   "--machine", "default", "-o", "grid.resolution=-5"])
    assert not out.exists()


def test_init_refuses_nonempty_output(mod, tmp_path):
    init, _ = mod
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "stuff.txt").write_text("x")
    with pytest.raises(SystemExit):
        init.main(["2d/williamson2_sw", "--name", "x", "--output-dir", str(out),
                   "--machine", "default"])
