"""Grid coverage for the coupled driver (B1): all four atmosphere grids are
selectable; coupled slab/ocean runs on cubed_sphere/latlon/voronoi AND gaussian
(spectral) — the spectral state is synthesized to grid in _build_atm_forcing and
_run_spectral recomputes/stashes the surface radiation at the coupling boundary
(A2). A 3-D dynamic ocean on gaussian runs on a DISTINCT lat-lon ocean grid via
the conservative cross-grid remap (GaussianGrid.lat_v quadrature edges), same as
the cube atm; a co-located spectral 3-D ocean stays idealized."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run.run_coupled import build_parser

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "run" / "run_coupled.py"

# JAX logs its own CUDA-plugin discovery failure WITH a traceback on a node
# without a GPU, even under JAX_PLATFORMS=cpu.  That record is not the
# driver's; drop it before asserting the driver raised no traceback.
_JAX_PLUGIN_NOISE = re.compile(
    r"[^\n]*Jax plugin configuration error[^\n]*\n"
    r"Traceback \(most recent call last\):\n(?:[ \t][^\n]*\n)*[^\n]*\n?")


def _driver_output(result) -> str:
    return _JAX_PLUGIN_NOISE.sub("", result.stdout + result.stderr)


@pytest.mark.parametrize("grid", ["cubed_sphere", "latlon", "voronoi", "gaussian"])
def test_grid_choice_accepted(grid):
    """All four grids parse (the --grid choices were widened for coupled runs)."""
    args = build_parser().parse_args(["--grid", grid])
    assert args.grid_type == grid


def test_gaussian_dynamic_ocean_requires_distinct_ocean_grid():
    """Coupled --grid gaussian --ocean dynamic WITHOUT --ocean-grid is a clean
    SystemExit (the spectral atm has no co-located 3-D ocean dycore; it needs a
    distinct lat-lon ocean grid), NOT a traceback and NOT the old blanket gate."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--preset", "full_coupled",
         "--grid", "gaussian", "--ocean", "dynamic", "--resolution", "8",
         "--days", "1"],
        env=env, capture_output=True, text=True, timeout=180)
    combined = _driver_output(result)
    assert result.returncode != 0, "gaussian+dynamic without --ocean-grid must exit"
    assert "gaussian" in combined and "--ocean-grid latlon" in combined, \
        f"missing the distinct-ocean-grid message.\n{combined[-600:]}"
    assert "Traceback" not in combined, \
        f"gate should be a clean SystemExit, not a traceback.\n{combined[-800:]}"


def test_gaussian_dynamic_ocean_with_distinct_grid_parses():
    """--grid gaussian --ocean dynamic --ocean-grid latlon:<res> is now REACHABLE
    (the Gaussian<->latlon conservative cross-grid remap is wired): the args
    parse and the distinct ocean-grid spec is accepted (no SystemExit at parse)."""
    args = build_parser().parse_args(
        ["--grid", "gaussian", "--ocean", "dynamic", "--ocean-grid", "latlon:48"])
    assert args.grid_type == "gaussian" and args.ocean == "dynamic"
    assert args.ocean_grid == "latlon:48"


# Child process for the gaussian/slab test: runs ``run_coupled.main`` unchanged
# (the full CLI path) and records the slab SST right after ``setup()`` and
# right after ``run()``.  The driver writes no SST field, so the change over the
# run is printed here for the test to parse.
_SST_CHANGE_CHILD = """
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import run_coupled
from legoesm.driver.coupled_esm_driver import CoupledESMDriver as D
rec = {}
def _wrap(name, key):
    orig = getattr(D, name)
    def f(self, *a, **k):
        out = orig(self, *a, **k)
        rec[key] = np.array(self._ocean_state.T_sfc.data, dtype=np.float64)
        rec["lat"] = np.asarray(self._atm._grid_lat, dtype=np.float64)
        return out
    setattr(D, name, f)
_wrap("setup", "sst0")
_wrap("run", "sst1")
run_coupled.main(sys.argv[2:])
d = rec["sst1"] - rec["sst0"]
lat = np.rad2deg(np.broadcast_to(rec["lat"], d.shape))
print("SST_CHANGE finite=%d tropics_mean=%.6e global_mean=%.6e"
      % (np.isfinite(d).all(), d[np.abs(lat) < 20.0].mean(), d.mean()))
"""


def test_gaussian_slab_coupled_runs(tmp_path):
    """Coupled --grid gaussian with a slab ocean RUNS end-to-end (A2) and the
    spectral lane's surface radiation reaches the slab (#1202, zero-SW class).

    Asserts the tropical-mean SST change over the 1-day run, not the SST
    spread (the initial analytic SST already spans 30 K).  Measured on CPU,
    fp64, unweighted mean over |lat| < 20: +0.089 K; with the shortwave into
    the ocean zeroed -0.003 K; with the ocean step skipped exactly 0.  The
    0.04 K bound is about half the measured value; re-measure it if the case
    (resolution, days, physics) changes."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _SST_CHANGE_CHILD, str(DRIVER.parent),
         "--grid", "gaussian", "--ocean", "slab",
         "--resolution", "16", "--days", "1", "--radiation", "gray",
         "--no-energy-fixer",   # not wired on the spectral lane (F15)
         # The faithful Kessler (#2083, the spectral default) runs a fixed
         # 256-pass sub-cycle loop every step: 1 day took ~12 min here vs ~4
         # without microphysics.  This test is about radiative forcing of the
         # ocean; Kessler has its own oracle tests.
         "--microphysics", "none",
         "--output", str(tmp_path / "coupled_gauss")],
        env=env, capture_output=True, text=True, timeout=600)
    combined = _driver_output(result)
    assert result.returncode == 0 and "COMPLETED" in combined, \
        f"coupled gaussian slab run did not complete.\n{combined[-1200:]}"
    assert "Traceback" not in combined and "nan" not in result.stdout.lower(), \
        f"coupled gaussian produced a traceback / NaN.\n{combined[-1200:]}"
    m = re.search(r"SST_CHANGE finite=(\d) tropics_mean=(\S+) global_mean=(\S+)",
                  combined)
    assert m, f"no SST change in output.\n{combined[-800:]}"
    assert m.group(1) == "1", "non-finite SST change"
    d_trop = float(m.group(2))
    assert d_trop > 0.04, (
        f"tropical-mean SST change {d_trop:.4f} K over 1 day; the slab is not "
        f"radiatively forced (zero-SW regression?) or not coupled")


def test_ocean_mode_label_mapping():
    """GAP-5 guard companion: the public accessor maps every SimpleOceanConfig
    mode onto its CoupledConfig.ocean_mode value, None on unknown."""
    from legoesm.driver.coupled_config import ocean_mode_label

    assert ocean_mode_label("fixed") == "slab"
    assert ocean_mode_label("slab") == "slab"
    assert ocean_mode_label("two_layer") == "two_layer"
    assert ocean_mode_label("dynamic") is None
    assert ocean_mode_label("typo") is None


def test_resolve_coupled_microphysics_per_grid_default():
    """2026-07-22 audit: coupled spectral path defaults to graph-tractable
    kessler (the double-moment default segfaults XLA-CPU codegen at production
    nlev); other grids keep morrison; explicit choices are honored."""
    from scripts.run.run_coupled import resolve_coupled_microphysics

    # default (None) is grid-dependent
    assert resolve_coupled_microphysics("gaussian", None) == (
        "kessler", "defaulted_kessler")
    for g in ("cubed_sphere", "latlon", "voronoi"):
        assert resolve_coupled_microphysics(g, None) == (
            "morrison", "defaulted_morrison")

    # explicit light scheme honored everywhere without warning
    assert resolve_coupled_microphysics("gaussian", "kessler") == (
        "kessler", "kept")
    assert resolve_coupled_microphysics("cubed_sphere", "morrison") == (
        "morrison", "kept")

    # explicit heavy scheme on spectral is honored but flagged for the warning
    assert resolve_coupled_microphysics("gaussian", "morrison") == (
        "morrison", "explicit_heavy_warn")
    # heavy scheme on a non-spectral grid is fine (no codegen wall there)
    assert resolve_coupled_microphysics("cubed_sphere", "morrison") == (
        "morrison", "kept")
