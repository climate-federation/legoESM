"""Unit tests for the Oceananigans -> legoESM C-grid state bridge.

Mirrors the MITgcm bridge tests, with the extra assertion that matters for
Oceananigans: the **vertical is reversed** (Oceananigans ``k=1`` is the BOTTOM,
``z`` increases upward; legoESM ``k=0`` is the SURFACE).
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.fidelity.oceananigans_runner import OceananigansResult
from legoesm.ocean.fidelity.oceananigans_state_bridge import (
    oceananigans_snapshot_to_legoesm_state,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star

N_LAT, N_LON = 6, 8


def _base_state(nlev):
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(nlev, H_max=500.0)
    return rest_state_latlon_cgrid_ocean(grid, z_coord, land_lat_threshold=90.0)


def _result(variables, *, cyclic_x=False, times_s=(0.0,)):
    return OceananigansResult(
        case_name="barotropic_gyre",
        times_s=np.asarray(times_s, dtype=np.float64),
        variables=variables,
        coords={},
        grid_metadata={"nx": N_LON, "ny": N_LAT, "nz": 1, "cyclic_x": cyclic_x},
        provenance={},
    )


def test_single_level_u_face_closed_wall():
    base = _base_state(1)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((N_LAT, N_LON))  # west-face values, single level (2-D)
    out = oceananigans_snapshot_to_legoesm_state(_result({"u": u}), base, cyclic_x=False)
    udata = np.asarray(out.state.u.data)
    assert udata.shape == base.u.data.shape  # (N_LAT, N_LON+1, 1)
    np.testing.assert_allclose(udata[:, :N_LON, 0], u)
    np.testing.assert_allclose(udata[:, N_LON, 0], 0.0)  # closed east wall
    assert out.info["placed_fields"] == ["u"]


def test_cyclic_x_wraps_u_east_face():
    base = _base_state(1)
    u = np.arange(N_LAT * N_LON, dtype=np.float64).reshape(N_LAT, N_LON)
    out = oceananigans_snapshot_to_legoesm_state(_result({"u": u}), base, cyclic_x=True)
    udata = np.asarray(out.state.u.data)
    np.testing.assert_allclose(udata[:, N_LON, 0], u[:, 0])  # periodic wrap


def test_v_face_north_wall():
    base = _base_state(1)
    v = np.arange(N_LAT * N_LON, dtype=np.float64).reshape(N_LAT, N_LON)
    out = oceananigans_snapshot_to_legoesm_state(_result({"v": v}), base)
    vdata = np.asarray(out.state.v.data)
    assert vdata.shape == base.v.data.shape  # (N_LAT+1, N_LON, 1)
    np.testing.assert_allclose(vdata[:N_LAT, :, 0], v)
    np.testing.assert_allclose(vdata[N_LAT, :, 0], 0.0)  # north wall


def test_eta_placed_directly():
    base = _base_state(1)
    eta = np.full((N_LAT, N_LON), 0.3)
    out = oceananigans_snapshot_to_legoesm_state(_result({"eta": eta}), base)
    assert out.state.eta.data.shape == base.eta.data.shape
    np.testing.assert_allclose(np.asarray(out.state.eta.data), eta)
    assert "eta" in out.info["placed_fields"]


def test_vertical_is_reversed():
    """Oceananigans bottom-up (k=0 bottom) -> legoESM surface-first (k=0 surface)."""
    nlev = 4
    base = _base_state(nlev)
    # Oceananigans u (z, y, x): z-index 0 = BOTTOM, z-index nlev-1 = SURFACE.
    # Encode the level in the value so the reversal is unambiguous.
    u = np.zeros((nlev, N_LAT, N_LON))
    for k in range(nlev):
        u[k, :, :] = float(k)  # 0 at bottom ... nlev-1 at surface
    out = oceananigans_snapshot_to_legoesm_state(_result({"u": u}), base)
    udata = np.asarray(out.state.u.data)  # (N_LAT, N_LON+1, nlev)
    # legoESM k=0 is the SURFACE -> must equal the oracle's TOP level (nlev-1).
    np.testing.assert_allclose(udata[:, :N_LON, 0], float(nlev - 1))
    # legoESM bottom level -> oracle's bottom (0).
    np.testing.assert_allclose(udata[:, :N_LON, nlev - 1], 0.0)


def test_already_has_boundary_face_passes_through():
    """If the oracle array already includes the boundary face (Bounded N+1),
    the adaptive face helper passes it through unchanged."""
    base = _base_state(1)
    u_full = np.arange(N_LAT * (N_LON + 1), dtype=np.float64).reshape(N_LAT, N_LON + 1)
    out = oceananigans_snapshot_to_legoesm_state(_result({"u": u_full}), base)
    udata = np.asarray(out.state.u.data)
    np.testing.assert_allclose(udata[:, :, 0], u_full)


def test_non_state_variable_skipped():
    """A variable with no canonical state mapping is recorded as skipped, not
    placed (and does not error)."""
    base = _base_state(1)
    out = oceananigans_snapshot_to_legoesm_state(
        _result({"eta": np.zeros((N_LAT, N_LON)), "some_diagnostic": np.zeros((N_LAT, N_LON))}),
        base)
    assert "eta" in out.info["placed_fields"]
    assert "some_diagnostic" in out.info["skipped_nonstate"]
