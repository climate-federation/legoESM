"""Unit test for the P-model EC scoring ladder aggregator."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import xarray as xr

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "validate" / "ec_pmodel_ladder.py"


def _write_site(path, rng, quality):
    n = 200
    obs = 10.0 + 5.0 * np.sin(np.arange(n) / 7.0)
    mod = obs + rng.normal(0.0, quality, n) + 0.5
    ds = xr.Dataset({
        "gpp_mod": ("time", mod), "gpp_obs": ("time", obs),
        "le_mod": ("time", 8.0 * mod), "le_obs": ("time", 8.0 * obs),
        "h_mod": ("time", 4.0 * mod), "h_obs": ("time", 4.0 * obs),
        "score_valid": ("time", np.ones(n, dtype="i1")),
    }, attrs={"pft": "DBF"})
    ds.to_netcdf(path)


def test_ladder_matched_sites_and_summary(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "arms"
    for arm, quality in (("a0", 3.0), ("a1", 1.0)):
        d = base / arm
        d.mkdir(parents=True)
        for site in ("S1", "S2"):
            _write_site(d / f"{site}_driver_v2.nc_ec_diagnostic.nc", rng,
                        quality)
    # a1 has an EXTRA site that must be dropped by the matched-site protocol
    _write_site(base / "a1" / "S3_driver_v2.nc_ec_diagnostic.nc", rng, 1.0)

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-dir", str(base),
         "--arms", "a0", "a1", "--parents", "none", "a0"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert "matched sites (2)" in out.stdout
    assert "dropped from matched set" in out.stdout

    md = (base / "ladder_summary.md").read_text()
    assert "| a1 | GPP |" in md
    csv = (base / "ladder_skill.csv").read_text()
    # 2 arms x 2 matched sites x 3 fluxes = 12 data rows
    assert len(csv.strip().splitlines()) == 13
    # the cleaner arm must show a positive median-r delta vs its parent
    a1_gpp = [ln for ln in md.splitlines() if ln.startswith("| a1 | GPP")][0]
    assert "+" in a1_gpp.split("|")[6]


def test_ladder_refuses_disjoint_arms(tmp_path):
    rng = np.random.default_rng(1)
    base = tmp_path / "arms"
    (base / "a0").mkdir(parents=True)
    (base / "a1").mkdir(parents=True)
    _write_site(base / "a0" / "S1_driver_v2.nc_ec_diagnostic.nc", rng, 1.0)
    _write_site(base / "a1" / "S2_driver_v2.nc_ec_diagnostic.nc", rng, 1.0)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-dir", str(base),
         "--arms", "a0", "a1", "--parents", "none", "a0"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode != 0
    assert "nothing comparable" in (out.stdout + out.stderr)
