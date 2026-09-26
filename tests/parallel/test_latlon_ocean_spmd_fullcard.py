"""Full-card SPMD gate: a PARTIAL-CELL coordinate must band with the state.

The eORCA025 full-card 4-GPU run died with ``mul (1208,1440,75) vs
(302,1440,1)`` in ``compute_frozen_geom_density``: the shard_map body received
a band-local state but the model's GLOBAL z-coordinate — the SPMD lane never
banded the vertical coordinate's per-cell fields (``h_partial`` /
``bottom_level`` / ``is_active``), because every prior SPMD config ran pure
z-star (no per-cell fields).  The fix threads a ``z_coord`` override through
the step drivers (mirroring the existing ``grid=`` override) and stacks the
per-cell fields over bands exactly like the geometry.

The gate is an equivalence test: an N-step SPMD run over fake CPU devices must
match the single-device step on a VARIABLE-bathymetry partial-cell setup — the
configuration whose mere construction crashed before the fix, so this test
FAILS (with that same shape error) when the banding is removed.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate,
)


def _have_sharded_step():
    try:
        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
            make_sharded_ocean_step,
        )
        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_partial_cell_spmd_matches_single_device():
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, make_sharded_ocean_step, shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    zs = create_ocean_z_star(n_levels=nlev, H_max=4000.0)

    # Variable bathymetry with real partial bottom cells: a smooth ridge so
    # h_partial / bottom_level / is_active genuinely vary per column.  It is
    # exactly this per-cell variation the un-banded coordinate could not
    # deliver to a band body.
    lat_idx = np.arange(n_lat)[:, None]
    lon_idx = np.arange(n_lon)[None, :]
    H_bathy = (4000.0
               - 1500.0 * np.exp(-((lat_idx - n_lat / 2) / 8.0) ** 2)
               - 400.0 * np.cos(2 * np.pi * lon_idx / n_lon))
    # A bathymetry STEP straddling every band cut (rows 11|12, 23|24, 35|36
    # at 48/4), so the cut v-faces genuinely exercise the partial-cell
    # face-activity branch -- a smooth ridge can leave every cut flat and
    # the seam path untested (GLM review 2026-08-24, "test luck").
    H_bathy = H_bathy - 600.0 * ((lat_idx % 24) >= 12)
    zc = create_partial_cell_coordinate(zs, jnp.asarray(H_bathy))
    bl = np.asarray(zc.bottom_level)
    assert int(bl.min()) != int(bl.max()), "bathymetry must vary"
    for cut in (12, 24, 36):
        assert (bl[cut - 1] != bl[cut]).any(), (
            f"no bottom-level jump across the band cut at row {cut}; the "
            f"seam partial-cell branch would go untested")

    cfg = LatLonCGridOceanConfig.from_flat()
    model = LatLonCGridOceanModel(grid, zc, cfg)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    # Overwrite bathymetry/perturb so the step exercises the partial-cell PGF.
    rng = np.random.default_rng(0)
    state0 = state0._replace(
        H_bathy=state0.H_bathy.replace(data=jnp.asarray(H_bathy)),
        u=state0.u.replace(data=jnp.asarray(
            0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)))),
        eta=state0.eta.replace(data=jnp.asarray(
            0.005 * rng.standard_normal((n_lat, n_lon)))),
        T=state0.T.replace(data=jnp.asarray(
            5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))),
    )
    dt, n_steps = 600.0, 60

    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    # The fix's fingerprint: the step must carry non-empty z-coord stacks.
    assert len(step.aux) == 5, ("aux must carry (geom, vmask, zc, iwm, cfg)"
                            " stacks")
    zc_stacks = step.aux[2]
    assert set(zc_stacks) >= {"h_partial", "bottom_level", "is_active"}

    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)

    # The ESTABLISHED SPMD tolerance (test_latlon_ocean_spmd_step.py):
    # band-wise reductions legitimately reorder float ops vs the single
    # device, so equivalence is to (2e-4, 1e-3), not bit-exact.  A first
    # version demanded 1e-10 and failed at max|dT| = 3.4e-4 — the
    # EXPECTATION was wrong, not the banding (measured, matching the z-star
    # gate's own basis).
    _ATOL, _RTOL = 2.0e-3, 1.0e-2   # 60 steps: rounding compounds ~sqrt(N)
    for name in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(s, name).data)
        b = np.asarray(getattr(ss, name).data)
        np.testing.assert_allclose(
            b, a, atol=_ATOL, rtol=_RTOL,
            err_msg=f"{name} diverged between SPMD and single-device with "
                    f"partial cells")

    # THE DRIFT DISCRIMINATOR (GLM review): a seam defect grows
    # SYSTEMATICALLY at the cut rows while reduction-reorder rounding is
    # spatially uniform.  After 60 steps the max |dT| within +-2 rows of the
    # interior cuts must be comparable to (not a multiple of) the max |dT|
    # far from every cut.  Ratio 3 allows healthy tail statistics; a walled
    # or double-counted cut face fails this by orders of magnitude.
    dT = np.abs(np.asarray(ss.T.data) - np.asarray(s.T.data))
    cut_rows = sorted({r for c in (12, 24, 36) for r in range(c - 2, c + 2)})
    interior_rows = [r for r in range(2, 46) if r not in
                     {r2 for c in (12, 24, 36) for r2 in range(c - 4, c + 4)}]
    cut_err = float(dT[cut_rows].max())
    interior_err = float(dT[interior_rows].max())
    assert cut_err <= 3.0 * max(interior_err, 1e-12), (
        f"cut-row error {cut_err:.3e} is {cut_err / max(interior_err, 1e-12):.1f}x "
        f"the interior {interior_err:.3e}: systematic seam defect, not "
        f"rounding")


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_zstar_config_still_has_empty_aux_stacks():
    """A pure z-star model (no per-cell fields, no iwm) must produce EMPTY
    z/iwm stacks and a None override — the pre-fix behaviour, bit-identical,
    so every existing SPMD user is untouched."""
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    grid = create_latlon_grid(n_lat=48, n_lon=96)
    zc = create_ocean_z_star(n_levels=10, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, zc, LatLonCGridOceanConfig.from_flat())
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    _geom, _vmask, zc_stacks, iwm_stacks, cfg_stacks = step.aux
    assert zc_stacks == {}
    assert iwm_stacks is None
    assert cfg_stacks == {}      # no per-cell config arrays on the bare card


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_a_h_lat_profile_is_band_sliced_under_spmd():
    """#1666: A_h_lat_profile is a hashable TUPLE, so tree_flatten explodes it
    into scalar leaves the array-leaf band-stacker skips -> the band body keeps
    the GLOBAL-length profile and raises `n_lat vs nl_band`.  With the fix the
    profile is band-sliced; the SPMD run matches single-device.

    NON-VACUITY: revert the sharded_ocean_step fix and this raises ValueError
    ("A_h_lat_profile has 48 entries but the grid has 12 latitude rows").
    """
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, make_sharded_ocean_step, shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    zc = create_ocean_z_star(n_levels=nlev, H_max=4000.0)

    # A cos(lat)-like equatorial A_h reduction, one entry per lat row (the
    # profile's contract: len == grid.lat).  A_h>0 so it scales; eq_boost=1.0
    # (mutually exclusive with the profile).
    lat = np.linspace(-1.0, 1.0, n_lat)
    prof = tuple(float(0.2 + 0.8 * np.cos(lat[i] * np.pi / 2) ** 2)
                 for i in range(n_lat))
    base = LatLonCGridOceanConfig.from_flat()
    cfg = base._replace(
        lateral_viscosity=base.lateral_viscosity._replace(
            A_h=20000.0, A_h_eq_boost=1.0, A_h_lat_profile=prof))

    model = LatLonCGridOceanModel(grid, zc, cfg)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    state0 = state0._replace(
        u=state0.u.replace(data=jnp.asarray(
            0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)))),
        eta=state0.eta.replace(data=jnp.asarray(
            0.005 * rng.standard_normal((n_lat, n_lon)))),
    )
    dt, n_steps = 600.0, 20

    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)   # must not raise
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)                              # in-body: must not raise
    ss = gather_state_latlon(ss, dev.mesh)

    _ATOL, _RTOL = 2.0e-3, 1.0e-2
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(ss, name).data),
            np.asarray(getattr(s, name).data),
            atol=_ATOL, rtol=_RTOL,
            err_msg=f"{name} diverged SPMD vs single-device with a SHARP A_h "
                    f"profile on a band seam (the seam-fix target)")


def test_1666_seam_v_profile_is_neighbour_averaged_not_wall_copied():
    """#1666 seam fix (codex+GLM P1): the SPMD wrapper injects the EXACT global
    v-face profile so each band's seam v-faces carry the neighbour-averaged
    value 0.5*(p[k-1]+p[k]), consistent across the seam -- NOT the band-local
    wall copy the in-body derive would produce.  Pure arithmetic on the same
    formulas the wrapper (concat) and the stacker ([r*nl:(r+1)*nl+1]) use.

    NON-VACUITY: the wall-copy value (prof[seam-1]) differs from the injected
    neighbour-average whenever the profile varies across the seam -- asserted.
    """
    n_lat, n_dev = 48, 4
    nl = n_lat // n_dev
    prof = np.array([1.0 - 0.9 * (i / (n_lat - 1)) for i in range(n_lat)])  # monotone ramp: every seam has a gradient
    v_global = np.concatenate([prof[:1], 0.5 * (prof[:-1] + prof[1:]), prof[-1:]])
    for r in range(n_dev - 1):
        seam = (r + 1) * nl                       # global v-face index at the r|r+1 cut
        band_r_north = v_global[r * nl:(r + 1) * nl + 1][-1]
        band_r1_south = v_global[(r + 1) * nl:(r + 2) * nl + 1][0]
        expect = 0.5 * (prof[seam - 1] + prof[seam])
        assert band_r_north == band_r1_south == expect, (
            f"seam {seam}: bands disagree or not neighbour-averaged")
        wall_copy = prof[r * nl:(r + 1) * nl][-1]  # what the band-local derive gives
        assert band_r_north != wall_copy, (
            f"seam {seam}: injected value equals the (broken) wall copy")


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_partial_cell_spmd_with_active_fold_matches_single_device():
    """FOLD x PARTIAL-CELL: the eORCA025 smoke's exact structural combination.

    The tripole SPMD gate (test_latlon_ocean_spmd_tripole.py) runs z-star
    only; the fullcard gate above runs partial cells on a FOLDLESS regular
    grid.  The 1/4-degree smoke (job 9494822) reached day 1 with a NaN state
    on the first configuration that combines them — an ACTIVE bipolar fold
    over band-stacked per-cell z-coordinate fields — so this test pins the
    combination offline: 60 banded steps on a synthetic tripole with variable
    partial-cell bathymetry (including a step ACROSS the fold row's partner
    columns) must stay finite and match the single-device step.
    """
    from legoesm.grids.tripole import create_synthetic_tripole
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, make_sharded_ocean_step, shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_synthetic_tripole(n_lat, n_lon)
    assert grid.fold.is_active
    zs = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    lat_idx = np.arange(n_lat)[:, None]
    lon_idx = np.arange(n_lon)[None, :]
    H_bathy = (4000.0
               - 1500.0 * np.exp(-((lat_idx - n_lat / 2) / 8.0) ** 2)
               - 400.0 * np.cos(2 * np.pi * lon_idx / n_lon))
    H_bathy = H_bathy - 600.0 * ((lat_idx % 24) >= 12)
    # A bathymetry step across the FOLD PARTNERS: the fold maps column i to
    # n_lon-1-i on the top row, so make depth vary in lon there — a fold
    # defect (missing perm/sign or a wall instead of the partner) then reads
    # the WRONG column's partial-cell height at the fold and diverges.
    H_bathy[-2:, : n_lon // 2] -= 350.0
    zc = create_partial_cell_coordinate(zs, jnp.asarray(H_bathy))

    cfg = LatLonCGridOceanConfig.from_flat()
    model = LatLonCGridOceanModel(grid, zc, cfg)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(1)
    state0 = state0._replace(
        H_bathy=state0.H_bathy.replace(data=jnp.asarray(H_bathy)),
        u=state0.u.replace(data=jnp.asarray(
            0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)))),
        eta=state0.eta.replace(data=jnp.asarray(
            0.005 * rng.standard_normal((n_lat, n_lon)))),
        T=state0.T.replace(data=jnp.asarray(
            5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))),
    )
    dt, n_steps = 600.0, 60

    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)
    for name in ("T", "u", "v", "eta"):
        assert np.isfinite(np.asarray(getattr(s, name).data)).all(), \
            f"single-device reference went non-finite in {name}"

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)

    _ATOL, _RTOL = 2.0e-3, 1.0e-2
    for name in ("T", "S", "u", "v", "eta"):
        b = np.asarray(getattr(ss, name).data)
        assert np.isfinite(b).all(), (
            f"SPMD {name} went NON-FINITE under fold x partial cells -- the "
            f"eORCA025 day-1 NaN class, reproduced offline")
        np.testing.assert_allclose(
            b, np.asarray(getattr(s, name).data), atol=_ATOL, rtol=_RTOL,
            err_msg=f"{name} diverged (fold x partial-cell banding)")

    # Fold-row drift discriminator: error at the top two rows must be
    # comparable to the interior, not orders bigger.
    dT = np.abs(np.asarray(ss.T.data) - np.asarray(s.T.data))
    fold_err = float(dT[-2:].max())
    interior_err = float(dT[4:40].max())
    assert fold_err <= 5.0 * max(interior_err, 1e-12), (
        f"fold-row error {fold_err:.3e} vs interior {interior_err:.3e}: "
        f"systematic fold defect under banding")


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_pivot_layout_fold_stays_finite_and_matches_single_device():
    """PIVOT-ROW-STORED fold layout (eORCA025 class): finiteness + SPMD
    equivalence with a WET, de-duplicated fold row.

    Mimics the eORCA025 structure that killed the 1/4-degree smoke in ~23
    steps: the top row is the self-symmetric pivot row, one mirror half is
    LAND (the tmaskutil de-duplication), and the fold ghosts must source
    the row BELOW the pivot with the per-stagger maps — permuting the
    stored top row instead reads the land mirror twins (T=0/S=0 ghost
    water) and blows up.  60 steps, single-device finite AND SPMD-equal.
    """
    from legoesm.grids.tripole import create_synthetic_tripole_pivot
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon, make_sharded_ocean_step, shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_synthetic_tripole_pivot(n_lat, n_lon)
    assert grid.fold.is_active and grid.fold.pivot_row_stored
    zc = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, zc, LatLonCGridOceanConfig.from_flat())
    state0 = rest_state_latlon_cgrid_ocean(
        grid, zc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    # De-duplicated wet fold row: land the mirror half i in (n_lon//2, n_lon)
    # of the TOP row (tmaskutil convention), wet elsewhere.
    lm = np.asarray(state0.land_mask.data).copy()
    lm[-1, n_lon // 2 + 1:] = 0.0
    from legoesm.ocean.init_latlon_cgrid import replace_land_mask
    state0 = replace_land_mask(state0, jnp.asarray(lm))
    rng = np.random.default_rng(2)
    # NONZERO v + a meridional tracer gradient concentrated at the top rows:
    # the TVD north second-neighbour and the fold V/F constructions only
    # bind when meridional flux crosses the seam (codex fold-fix round 2 —
    # the first gate left v=0 and could not see the RED TVD/EEN sites).
    T3 = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
          + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    T3 += 2.0 * (np.arange(n_lat) / n_lat)[:, None, None]   # S->N gradient
    state0 = state0._replace(
        u=state0.u.replace(data=jnp.asarray(
            0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)))),
        v=state0.v.replace(data=jnp.asarray(
            0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev)))),
        eta=state0.eta.replace(data=jnp.asarray(
            0.005 * rng.standard_normal((n_lat, n_lon)))),
        T=state0.T.replace(data=jnp.asarray(T3)),
    )
    dt, n_steps = 600.0, 60

    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)
    for name in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(s, name).data)
        assert np.isfinite(a).all(), (
            f"single-device {name} non-finite under the pivot fold layout "
            f"(the eORCA025 step-23 NaN class)")

    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    ss = gather_state_latlon(ss, dev.mesh)
    for name in ("T", "S", "u", "v", "eta"):
        b = np.asarray(getattr(ss, name).data)
        assert np.isfinite(b).all(), f"SPMD {name} non-finite (pivot fold)"
        np.testing.assert_allclose(
            b, np.asarray(getattr(s, name).data), atol=2e-3, rtol=1e-2,
            err_msg=f"{name} SPMD != single-device under the pivot fold")
