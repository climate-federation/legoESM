"""The mock-vs-real gap closer for the operator-split lat-band SPMD lane.

Phase 2b-main validated the sharded operator-split STEP MECHANICS with a
column-local MOCK ``step_unified``.  This test closes the one remaining
assumption everything rests on: that the REAL ``PhysicsPipeline.step_unified``
(gray radiation + prognostic-TKE turbulence) is itself PURELY COLUMN-LOCAL, so a
lat-band decomposition is bit-exact.

Decisive check (no full carry/statics/forcing harness needed): build the REAL
``step_unified`` on the FULL ``(n_lat, n_lon)`` grid AND on a single
``(nl, n_lon)`` lat BAND, feed the band the SAME columns (the top ``nl`` rows of
the global inputs), and assert every ``PhysicsOutput`` field for the band equals
the global output's band slice.  If ANY field differs, the pipeline has a
horizontal (non-column-local) op and the SPMD lane would silently diverge —
identify it; do NOT loosen the tolerance.

Host CPU (``XLA_FLAGS=--xla_force_host_platform_device_count>=1``); needs no
extra devices (this is a single-process numerical-locality check).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    build_band_grids_atm)

N_LAT = 8
N_LON = 8
NLEV = 5
N_DEV = 2
NL = N_LAT // N_DEV


def _experiment(nlev=NLEV):
    """Minimal lat-lon full-physics ExperimentConfig: gray radiation (column,
    deterministic) + prognostic-TKE turbulence (the stateful carry) + no
    convection/microphysics. Cribbed from test_feedback_column_ordering."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="tke", convection="none",
        microphysics="none", gravity_wave_drag="none",
    )


def _grid_lat_lon(grid):
    """(lat, lon) coordinate arrays the pipeline's _single_step passes to
    step_unified — the grid's own grid_lat / grid_lon (band-sliced for a band)."""
    return jnp.asarray(grid.grid_lat), jnp.asarray(grid.grid_lon)


def _inputs(nlat, nlon, nlev, rng):
    """Grid-shaped step_unified inputs for an (nlat, nlon) grid. Deterministic +
    lat/lon-varying so a horizontal op would show up as a band mismatch."""
    eps = 1e-3
    T = jnp.asarray(280.0 + 20.0 * rng.standard_normal((nlat, nlon, nlev)) * eps
                    + 5.0 * np.linspace(0, 1, nlat)[:, None, None])
    p_s = jnp.asarray(1.0e5 + 100.0 * rng.standard_normal((nlat, nlon)))
    q_v = jnp.asarray(0.01 + 0.002 * rng.standard_normal((nlat, nlon, nlev)))
    q_c = jnp.zeros((nlat, nlon, nlev))
    q_r = jnp.zeros((nlat, nlon, nlev))
    u = jnp.asarray(5.0 + eps * rng.standard_normal((nlat, nlon, nlev)))
    v = jnp.asarray(eps * rng.standard_normal((nlat, nlon, nlev)))
    sst = jnp.full((nlat, nlon), 300.0)
    sic = jnp.zeros((nlat, nlon))
    o3 = jnp.zeros((nlat, nlon, nlev))
    aod = jnp.zeros((nlat, nlon))
    z2 = jnp.zeros((nlat, nlon))
    z3 = jnp.zeros((nlat, nlon, nlev))
    tke = jnp.full((nlat * nlon, nlev), 1e-4)
    return dict(T=T, p_s=p_s, q_v=q_v, q_c=q_c, q_r=q_r, u=u, v=v, sst=sst,
                sic=sic, o3=o3, aod=aod, held3=z3, held2=z2, tke=tke)


def _call(su, grid, inp):
    lat, lon = _grid_lat_lon(grid)
    return su(
        jnp.bool_(True),
        inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"], None,
        inp["u"], inp["v"], inp["sst"], inp["sic"], lat, lon,
        1.0, 0.0, 600.0,
        jnp.ones(14), constants.S_0, inp["o3"], inp["aod"],
        inp["held3"], inp["held2"], inp["held2"],
        inp["held2"], inp["held2"], inp["held2"],
        tke=inp["tke"],
    )


def _band_slice(inp):
    """The top NL lat rows of the global inputs — the band-0 columns. Grid-shaped
    leaves slice [:NL]; the flattened (ncol, nlev) tke slices [:NL*N_LON] (the
    lat-major first NL rows)."""
    out = {}
    for k, v in inp.items():
        if k == "tke":
            out[k] = v[:NL * N_LON]
        else:
            out[k] = v[:NL]
    return out


def test_real_step_unified_is_column_local():
    """REAL step_unified (gray rad + TKE) on a band == on the full grid, band
    slice — proving the pipeline is column-local so the operator-split SPMD lane
    is bit-exact with real physics (closing the 2b-main mock gap)."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
                              omega=constants.Omega)
    band = build_band_grids_atm(grid, N_DEV)[0]      # south band, NL rows
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = _experiment()

    su_g = build_physics_pipeline(grid, sigma, cfg).build_step_unified(
        static_need_rad=True, jit=True)
    su_b = build_physics_pipeline(band, sigma, cfg).build_step_unified(
        static_need_rad=True, jit=True)

    rng = np.random.default_rng(4242)
    inp = _inputs(N_LAT, N_LON, NLEV, rng)

    ret_g = _call(su_g, grid, inp)
    ret_b = _call(su_b, band, _band_slice(inp))

    phys_g, held_g = ret_g[0], ret_g[1]
    phys_b, held_b = ret_b[0], ret_b[1]

    # Every PhysicsOutput array field: the band's value must equal the global
    # value's top-NL-row slice (grid-shaped) or top-NL*N_LON slice (flattened).
    worst = {}
    for name in phys_g._fields:
        a = getattr(phys_g, name)
        b = getattr(phys_b, name)
        if a is None or b is None:
            assert a is None and b is None, f"{name}: None mismatch"
            continue
        a = np.asarray(a.data if hasattr(a, "data") else a)
        b = np.asarray(b.data if hasattr(b, "data") else b)
        if a.ndim == 0:
            continue
        # flattened (ncol, ...) fields (tke, conv_prog) slice lat-major;
        # grid-shaped (n_lat, ...) slice the first NL rows.
        ref = a[:NL * N_LON] if a.shape[0] == N_LAT * N_LON else a[:NL]
        assert ref.shape == b.shape, f"{name}: shape {b.shape} vs slice {ref.shape}"
        worst[name] = float(np.max(np.abs(b - ref))) if b.size else 0.0

    # held radiation tuple (6 arrays) — same locality contract.
    for i, (hg, hb) in enumerate(zip(held_g, held_b)):
        hg, hb = np.asarray(hg), np.asarray(hb)
        ref = hg[:NL * N_LON] if hg.shape[0] == N_LAT * N_LON else hg[:NL]
        worst[f"held[{i}]"] = float(np.max(np.abs(hb - ref)))

    bad = {k: w for k, w in worst.items() if w > 1e-10}
    assert not bad, (
        f"REAL step_unified is NOT column-local — band != global band-slice for "
        f"{bad}. A horizontal (non-column-local) physics op means the "
        f"operator-split SPMD lane would silently diverge from serial.")

    # Non-vacuity: the physics actually did something (gray rad heats T).
    dT = np.asarray(phys_g.dT_dt.data if hasattr(phys_g.dT_dt, "data")
                    else phys_g.dT_dt)
    assert float(np.max(np.abs(dT))) > 0.0, "physics inert — vacuous locality"
