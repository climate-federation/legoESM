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


@pytest.mark.parametrize("grid", ["cubed_sphere", "latlon", "voronoi", "gaussian"])
def test_grid_choice_accepted(grid):
    """All four grids parse (the --grid choices were widened for coupled runs)."""
    args = build_parser().parse_args(["--grid", grid])
    assert args.grid == grid


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
    combined = result.stdout + result.stderr
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
    assert args.grid == "gaussian" and args.ocean == "dynamic"
    assert args.ocean_grid == "latlon:48"


def test_gaussian_slab_coupled_runs(tmp_path):
    """Coupled --grid gaussian with a slab ocean RUNS end-to-end (A2): the
    spectral atm state is synthesized to grid in _build_atm_forcing, and
    _run_spectral recomputes + stashes the surface net radiation at the daily
    coupling boundary so the ocean is radiatively forced (not the zero-SW
    #1202-class bug). The equator-to-pole SST structure that develops is the
    tell-tale of nonzero insolation-driven forcing."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--grid", "gaussian", "--ocean", "slab",
         "--resolution", "16", "--days", "1", "--radiation", "gray",
         "--output", str(tmp_path / "coupled_gauss")],
        env=env, capture_output=True, text=True, timeout=600)
    combined = result.stdout + result.stderr
    assert result.returncode == 0 and "COMPLETED" in combined, \
        f"coupled gaussian slab run did not complete.\n{combined[-1200:]}"
    assert "Traceback" not in combined and "nan" not in result.stdout.lower(), \
        f"coupled gaussian produced a traceback / NaN.\n{combined[-1200:]}"
    # Nonzero radiative forcing => the ocean SST develops an equator-to-pole
    # spread (a zero-SW-forced slab would stay flat at its init).
    m = re.search(r"SST final \(ocean\): mean=[\d.]+K, range=\[([\d.]+), "
                  r"([\d.]+)\]K", combined)
    assert m, f"no SST range in output.\n{combined[-800:]}"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert hi - lo > 5.0, (
        f"SST spread {hi - lo:.1f}K too small — ocean under-forced (zero-SW "
        f"regression?)")


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
