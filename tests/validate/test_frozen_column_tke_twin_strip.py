"""frozen_column_tke_twin strip mode: the nemo_z0 surface level is refused by
default (control/nemo modes) and accepted only for the direct-K strip mode,
and the strip CLI requires its hourly-stress inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"
sys.path.insert(0, str(_DIR))
import frozen_column_tke_twin as twin  # noqa: E402


def _manifest(tmp_path, level):
    tke = {"c_k": 0.1, "tke_surface_bc_level": level, "n2_mode": "nemo_bn2"}
    m = {"config": {"resolved_config": {"runtime_config": {
        "physics": {"vertical_mixing": {"tke": tke}},
        "eos": "teos10", "constants": {"rho_0": 1026.0, "g": 9.80665}}}}}
    p = tmp_path / "run_manifest.json"
    p.write_text(json.dumps(m))
    return p


def test_nemo_z0_refused_unless_direct_strip(tmp_path):
    p = _manifest(tmp_path, "nemo_z0")
    with pytest.raises(SystemExit, match="tke_surface_bc_level=nemo_z0"):
        twin.load_resolved_config(p)
    cfg, eos, rho0, g = twin.load_resolved_config(p, allow_z0_direct=True)
    assert cfg.tke_surface_bc_level == "nemo_z0" and eos == "teos10"


def test_strip_mode_requires_hourly_stress():
    with pytest.raises(SystemExit, match="taum-file"):
        twin.main(["--mode", "strip", "--manifest", "m.json",
                   "--restart-glob", "x*", "--nemo-meshmask", "mm.nc"])


def test_ratio_class_is_symmetric_with_inconclusive_band():
    assert twin.ratio_class(1.0) == "within"
    assert twin.ratio_class(1 / 1.19) == "within"
    assert twin.ratio_class(1.5) == twin.ratio_class(1 / 1.5) == "INCONCLUSIVE"
    assert twin.ratio_class(3.3) == twin.ratio_class(0.25) == "OUTSIDE"
    assert twin.ratio_class(float("nan")) == "OUTSIDE"
