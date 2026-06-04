"""Item 4 Day 4 — stability smoke for the tropical-OMIP forcing path.

Runs ``_jra55_step`` for many consecutive steps against a synthetic
JRA55-do cache and verifies the state stays finite and physically
sensible. This is the integration-level gate that the per-step Day-2
and Day-3 unit tests can't catch:

- Closure features must compose cleanly across many steps.
- The Python time loop must not introduce JIT-recompilation per step.
- Surface fluxes (τ, Q_net, P-E) must stay in plausible magnitude
  ranges across the run.
- Top-layer T must remain bounded by physical limits.

The test runs 96 steps at dt=300 s = 8 simulated hours. That covers
several 3-hourly cache slots so the time-interpolation logic is
exercised, but stays cheap enough for CI (~10 s wall time on a
dev machine).

The full 30-day production smoke documented in
``forced_ocean_driver_plan.md`` requires real JRA55-do data and lives
outside CI. The bottom of this file documents how to run it.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

# Reuse the helpers from the Day-2/3 dispatch tests.
_DAY23 = (
    Path(__file__).resolve().parent / "test_run_omip_jra55_dispatch.py"
)
_spec = importlib.util.spec_from_file_location("_day23", _DAY23)
_day23 = importlib.util.module_from_spec(_spec)
sys.modules["_day23"] = _day23
_spec.loader.exec_module(_day23)

run_omip = _day23.run_omip
_make_synthetic_cache = _day23._make_synthetic_cache
_make_tiny_latlon_setup = _day23._make_tiny_latlon_setup
_argparse_namespace = _day23._argparse_namespace
_make_woa_like_targets = _day23._make_woa_like_targets


jax.config.update("jax_enable_x64", True)


# ============================================================================
# Multi-step stability smoke
# ============================================================================

def _build_smoke_run(tmp_path: Path, n_records: int = 200,
                    n_lat: int = 8, n_lon: int = 16):
    """Set up a tiny stability-smoke run against a synthetic cache.

    n_records=200 gives 25 simulated days of 3-hourly forcing — far
    more than the 96 steps × dt=300 s = 8 hours we actually run.
    """
    cache = _make_synthetic_cache(
        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=n_records,
    )
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    jra55_state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )
    return model, state, jra55_state


@pytest.mark.timeout(120)
def test_jra55_step_stable_for_96_consecutive_steps(tmp_path):
    """8 simulated hours under JRA55-do bulk-flux forcing — full
    closure-domain pipeline (sponge + SSS restoring + freeze cap)."""
    model, state, jra55_state = _build_smoke_run(tmp_path)
    dt = 300.0
    n_steps = 96
    t0 = time.time()
    for i in range(n_steps):
        state = run_omip._jra55_step(state, i, dt, model, jra55_state)
        # Periodically check finiteness so failure is surfaced early.
        if (i + 1) % 24 == 0:
            assert bool(jnp.all(jnp.isfinite(state.T.data))), (
                f"non-finite T at step {i+1}"
            )
            assert bool(jnp.all(jnp.isfinite(state.S.data))), (
                f"non-finite S at step {i+1}"
            )
            assert bool(jnp.all(jnp.isfinite(state.eta.data))), (
                f"non-finite eta at step {i+1}"
            )
    elapsed = time.time() - t0

    # Final state passes physical-bounds sanity:
    T_final = np.asarray(state.T.data)
    S_final = np.asarray(state.S.data)
    eta_final = np.asarray(state.eta.data)

    # Surface T in [-3, 45] °C is a generous physical bound. State T
    # is stored in °C on the lat-lon C-grid, so no conversion needed.
    T_top_C = T_final[..., 0]
    assert float(np.nanmin(T_top_C)) > -3.0, "surface T below -3°C"
    assert float(np.nanmax(T_top_C)) < 45.0, "surface T above 45°C"

    # Salinity in [20, 45] PSU is a generous bound.
    S_top = S_final[..., 0]
    assert float(np.nanmin(S_top)) > 20.0, "surface S below 20 PSU"
    assert float(np.nanmax(S_top)) < 45.0, "surface S above 45 PSU"

    # SSH within ±5 m for stability — large excursions imply CFL-like
    # numerical issues.
    assert float(np.nanmax(np.abs(eta_final))) < 5.0, "|eta| > 5 m"

    # The Python loop should not be pathologically slow on a tiny
    # grid: 96 steps in under 90 seconds is a generous budget.
    assert elapsed < 90.0, f"loop took {elapsed:.1f} s, expected < 90 s"


@pytest.mark.timeout(120)
def test_jra55_step_stable_with_features_disabled(tmp_path):
    """Stability is robust even when closure features are off — covers
    the bare-bulk-flux path so a regression that only shows up under
    one closure path (e.g. SSS restoring numerically blows up at a
    high piston velocity) doesn't slip through."""
    cache = _make_synthetic_cache(
        tmp_path, n_lat=8, n_lon=16, n_records=200,
    )
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(n_lat=8, n_lon=16)
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(
        jra55_cache=str(cache),
        jra55_no_sponge=True,
        jra55_no_sss_restoring=True,
        jra55_no_freeze_cap=True,
    )
    jra55_state = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )

    dt = 300.0
    for i in range(48):
        state = run_omip._jra55_step(state, i, dt, model, jra55_state)
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.S.data)))
    assert bool(jnp.all(jnp.isfinite(state.eta.data)))


# ============================================================================
# Surface flux magnitudes are physically plausible
# ============================================================================

def test_one_step_flux_magnitudes_are_realistic(tmp_path):
    """One step under synthetic forcing should produce surface fluxes
    in the expected sign and magnitude band:

    - |τ_x|, |τ_y| ≲ 1 Pa (peaks in storm tracks ~0.3 Pa; synthetic
      forcing has |U_a| ≲ 5 m/s so we expect smaller).
    - |Q_net| ≲ 1000 W/m² (instantaneous; cycle-1 mean should be
      within ±50 W/m², but instantaneous can spike).
    - Evap ~ 1e-5–1e-4 kg/m²/s for L_h ~ 25-250 W/m².

    A regression that produces 100x too-large stress (e.g. broken
    LY09 fix from Item 1) or zero stress (e.g. broken AtmToSurface
    from Day 1) would fail this test.
    """
    from legoesm import constants
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.forcing.jra55_do import (
        jra55_to_atm_surface, jra55_to_freshwater, load_jra55_slice,
    )

    model, state, jra55_state = _build_smoke_run(tmp_path, n_records=8)
    slc = load_jra55_slice(jra55_state["cache_path"], day=0.0)
    atm = jra55_to_atm_surface(
        slc, jra55_state["lat_2d"], jra55_state["lon_2d"], day=0.0,
    )
    # State T is in °C — convert to K for the bulk-flux solver.
    sst_K = state.T.data[..., 0] + constants.T_freeze
    tile = ocean_tile_response(
        atm, sst_K, jnp.zeros_like(sst_K), jnp.zeros_like(sst_K),
        jra55_state["coupler_cfg"],
    )

    tau_x = np.asarray(tile.tau_x)
    tau_y = np.asarray(tile.tau_y)
    sh = np.asarray(tile.shflx)
    lh = np.asarray(tile.lhflx)

    assert float(np.max(np.abs(tau_x))) < 5.0, "|tau_x| > 5 Pa unphysical"
    assert float(np.max(np.abs(tau_y))) < 5.0, "|tau_y| > 5 Pa unphysical"
    assert float(np.max(np.abs(sh))) < 1000.0, "|SH| > 1000 W/m² unphysical"
    assert float(np.max(np.abs(lh))) < 2000.0, "|LH| > 2000 W/m² unphysical"
    # Stress must not be uniformly zero (would indicate broken bulk flux).
    assert float(np.max(np.abs(tau_x)) + np.max(np.abs(tau_y))) > 1e-4

    # Freshwater fluxes:
    fw = jra55_to_freshwater(slc, tile.lhflx)
    evap = np.asarray(fw.evap)
    precip = np.asarray(fw.precip)
    assert float(np.max(np.abs(evap))) < 1e-3, "|evap| > 1e-3 kg/m²/s unphysical"
    assert float(np.max(np.abs(precip))) < 1e-3


# ============================================================================
# Real-data 30-day production smoke (manual, NOT in CI)
# ============================================================================

# The full 30-day production smoke from forced_ocean_driver_plan.md
# requires JRA55-do v1.4+ corrected data, which is not present in the
# repo and can't be downloaded from CI. Documenting the recipe here so
# it's discoverable.
#
# Pre-requisites:
#   1. JRA55-do v1.4+ corrected files staged locally as a single Zarr,
#      or as a directory of NetCDF files xarray.open_mfdataset can read.
#   2. WOA18 T/S NetCDFs (already used by run_omip.py restoring path).
#
# Build the cache once:
#
#   python scripts/data/prepare_omip_forcing.py \
#       --source /scratch/jra55_do_v14/raw.zarr \
#       --years 2000 2000 \
#       --target-resolution-deg 1.0 \
#       --cache-dir /scratch/legoESM/jra55_do_smoke
#
# Then run the smoke at 1°/30 days:
#
#   JAX_ENABLE_X64=1 python scripts/run_omip.py \
#       --grid latlon \
#       --resolution 180x360 \
#       --nlev 20 \
#       --dt 300 \
#       --days 30 \
#       --forcing-mode jra55_do_tropical \
#       --jra55-cache /scratch/legoESM/jra55_do_smoke/jra55_do_v14_omip2_1deg_noleap.zarr \
#       --woa-t /path/to/woa18_t.nc \
#       --woa-s /path/to/woa18_s.nc \
#       --output results/jra55_smoke_30d
#
# Expected wall time: ~30-60 minutes on a workstation GPU at 1°/dt=300s.
# Success criteria from forced_ocean_driver_plan.md §7:
#   - State remains finite for the full 30 days (no NaN/Inf).
#   - Daily diagnostics show realistic surface stress (peak ~0.3 Pa in
#     storm tracks under real winds), global-mean Q_net within ±10 W/m²,
#     and MLD evolution that broadly tracks seasonal forcing.
#   - AMOC@26.5N is order-of-magnitude correct (a few Sv to ~20 Sv).
