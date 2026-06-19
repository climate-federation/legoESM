"""Unit tests for the MITgcm -> legoESM C-grid state bridge."""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.fidelity.mitgcm_runner import MitgcmResult
from legoesm.ocean.fidelity.mitgcm_state_bridge import (
    mitgcm_snapshot_to_legoesm_state,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star

N_LAT, N_LON = 6, 8


def _base_state(nlev):
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(nlev, H_max=500.0)
    return rest_state_latlon_cgrid_ocean(grid, z_coord, land_lat_threshold=90.0)


def _result(variables, *, iters=(10,)):
    return MitgcmResult(
        case_name="barotropic_gyre",
        iters=np.asarray(iters, dtype=np.int64),
        times_s=np.asarray(iters, dtype=np.float64) * 1200.0,
        variables=variables,
        grid_metadata={"nx": N_LON, "ny": N_LAT, "nz": 1, "delta_t_s": 1200.0},
        provenance={},
    )


def test_single_level_u_face_placement_closed_wall():
    base = _base_state(1)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((N_LAT, N_LON)).astype(np.float64)  # west-face values
    out = mitgcm_snapshot_to_legoesm_state(_result({"U": u}), base, cyclic_x=False)
    udata = np.asarray(out.state.u.data)
    assert udata.shape == base.u.data.shape  # (N_LAT, N_LON+1, 1)
    # West faces land 1:1; the extra east face is the closed wall (zero).
    np.testing.assert_allclose(udata[:, :N_LON, 0], u)
    np.testing.assert_allclose(udata[:, N_LON, 0], 0.0)
    assert out.info["placed_fields"] == ["u"]


def test_single_level_v_face_placement_north_wall():
    base = _base_state(1)
    v = np.arange(N_LAT * N_LON, dtype=np.float64).reshape(N_LAT, N_LON)
    out = mitgcm_snapshot_to_legoesm_state(_result({"V": v}), base)
    vdata = np.asarray(out.state.v.data)
    assert vdata.shape == base.v.data.shape  # (N_LAT+1, N_LON, 1)
    np.testing.assert_allclose(vdata[:N_LAT, :, 0], v)
    np.testing.assert_allclose(vdata[N_LAT, :, 0], 0.0)  # north wall


def test_eta_placed_directly():
    base = _base_state(1)
    eta = np.full((N_LAT, N_LON), 0.3)
    out = mitgcm_snapshot_to_legoesm_state(_result({"Eta": eta}), base)
    # eta is a 2-D surface field (n_lat, n_lon) — no vertical axis.
    assert out.state.eta.data.shape == base.eta.data.shape
    np.testing.assert_allclose(np.asarray(out.state.eta.data), eta)
    assert "eta" in out.info["placed_fields"]


def test_cyclic_x_wraps_u_east_face():
    base = _base_state(1)
    u = np.arange(N_LAT * N_LON, dtype=np.float64).reshape(N_LAT, N_LON)
    out = mitgcm_snapshot_to_legoesm_state(_result({"U": u}), base, cyclic_x=True)
    udata = np.asarray(out.state.u.data)
    # Periodic: the wrap face equals column 0, not a zero wall.
    np.testing.assert_allclose(udata[:, N_LON, 0], u[:, 0])


def test_omitted_tracers_left_at_base():
    base = _base_state(1)
    # Homogeneous gyre snapshot: only U/V/Eta, no T/S.
    out = mitgcm_snapshot_to_legoesm_state(
        _result({"U": np.zeros((N_LAT, N_LON)), "V": np.zeros((N_LAT, N_LON))}),
        base,
    )
    np.testing.assert_array_equal(out.state.T.data, base.T.data)
    np.testing.assert_array_equal(out.state.S.data, base.S.data)
    assert "T" in out.info["left_at_base"] and "S" in out.info["left_at_base"]


def test_3d_axis_reorder_and_placement():
    nlev = 3
    base = _base_state(nlev)
    # MITgcm 3-D order: (nz, ny, nx). Encode the index so reorder is checkable.
    u = np.empty((nlev, N_LAT, N_LON))
    for k in range(nlev):
        for j in range(N_LAT):
            for i in range(N_LON):
                u[k, j, i] = 100 * k + 10 * j + i
    t = np.full((nlev, N_LAT, N_LON), 7.0)
    out = mitgcm_snapshot_to_legoesm_state(_result({"U": u, "Theta": t}), base)
    udata = np.asarray(out.state.u.data)
    assert udata.shape == base.u.data.shape  # (N_LAT, N_LON+1, nlev)
    # legoESM [j, i, k] must equal MITgcm [k, j, i] for the interior faces.
    for k in range(nlev):
        for j in range(N_LAT):
            for i in range(N_LON):
                assert udata[j, i, k] == 100 * k + 10 * j + i
    np.testing.assert_allclose(np.asarray(out.state.T.data), 7.0)


def test_multi_iteration_time_index_selects_snapshot():
    base = _base_state(1)
    # Two stacked snapshots; eta encodes the iteration.
    eta = np.stack([np.full((N_LAT, N_LON), 1.0), np.full((N_LAT, N_LON), 2.0)])
    res = _result({"Eta": eta}, iters=(10, 20))
    out_last = mitgcm_snapshot_to_legoesm_state(res, base, time_index=-1)
    np.testing.assert_allclose(np.asarray(out_last.state.eta.data), 2.0)
    assert out_last.info["iteration"] == 20
    out_first = mitgcm_snapshot_to_legoesm_state(res, base, time_index=0)
    np.testing.assert_allclose(np.asarray(out_first.state.eta.data), 1.0)
    assert out_first.info["iteration"] == 10


def test_uniform_field_maps_to_uniform_interior():
    """Equivariance sanity: a constant snapshot stays constant on the interior."""
    base = _base_state(1)
    c = 0.42
    out = mitgcm_snapshot_to_legoesm_state(
        _result({"U": np.full((N_LAT, N_LON), c)}), base
    )
    np.testing.assert_allclose(np.asarray(out.state.u.data)[:, :N_LON, 0], c)


def test_unconformable_shape_raises():
    base = _base_state(1)
    bad = np.zeros((N_LAT + 3, N_LON))  # wrong ny
    with pytest.raises(ValueError, match="cannot be conformed"):
        mitgcm_snapshot_to_legoesm_state(_result({"Eta": bad}), base)


def test_single_iteration_ignores_out_of_range_time_index():
    """time_index is ignored for a single-iteration result (no IndexError)."""
    base = _base_state(1)
    res = _result({"Eta": np.full((N_LAT, N_LON), 0.5)}, iters=(10,))
    out = mitgcm_snapshot_to_legoesm_state(res, base, time_index=5)
    assert out.info["iteration"] == 10
    np.testing.assert_allclose(np.asarray(out.state.eta.data), 0.5)


def test_duplicate_canonical_alias_raises():
    """A snapshot carrying two aliases for one field (T and Theta) is rejected."""
    base = _base_state(1)
    res = _result(
        {"T": np.zeros((N_LAT, N_LON)), "Theta": np.ones((N_LAT, N_LON))}, iters=(10,)
    )
    with pytest.raises(ValueError, match="same legoESM state field"):
        mitgcm_snapshot_to_legoesm_state(res, base)
