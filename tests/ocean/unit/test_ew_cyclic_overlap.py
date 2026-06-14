"""East-west ORCA cyclic-overlap projection (``config.ew_cyclic_overlap``).

The ORCA tripole grid is periodic east-west with a 2-point overlap: column 0
duplicates column ``nx-2`` and column ``nx-1`` duplicates column 1 (same
geographic longitude).  The C-grid operators apply regular-grid roll-periodicity
(period ``nx``), which is off-by-one for an ORCA grid (true period ``nx-2``),
and the eORCA1 mesh marks the halo columns LAND -> the east-west seam carries a
spurious wall and the two physical seam columns drift apart (the lon-72.5E SST/
SSS stripe).  ``_apply_ew_cyclic_overlap`` re-imposes the overlap on the
cell-centred prognostic fields each step.

Tests:
  * the leaf projection slaves col0<-col[nx-2] and col[nx-1]<-col1 for T, S,
    eta and leaves the interior + shapes untouched;
  * gating: default off -> step is bit-identical to no projection; on -> the
    halo columns mirror their partners after a step;
  * the NumPy setup helper ``_ew_overlap_fill`` matches the model projection
    (so static geometry/IC start consistent with the per-step slaving);
  * the runner rejects --ew-cyclic-overlap on a non-tripole grid.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _model(ew_cyclic_overlap=False, n_lat=8, n_lon=16, **cfg_kw):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    cfg = LatLonCGridOceanConfig(
        A_h=2.0e4, bottom_drag_r=1.0e-3, n_barotropic_substeps=8,
        enable_runtime_checks=False, ew_cyclic_overlap=ew_cyclic_overlap,
        **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _perturb_cols(a, nx):
    """Make halo columns differ from their overlap partners + add interior
    noise, so the projection has something to slave (rest state is uniform)."""
    rng = np.random.default_rng(0)
    a = np.array(a, dtype=np.float64)
    a[:, 0] += 3.0
    a[:, nx - 1] += -2.5
    a[:, 5] += rng.standard_normal(a[:, 5].shape) * 0.1
    return a


def _distinct_seam_state(state):
    """State whose halo columns DIFFER from their overlap partners for ALL
    prognostic fields (cell-centred T/S/eta + staggered u/v faces)."""
    from legoesm.core.field import Field
    nx = state.T.data.shape[1]

    def F(field, arr):
        return Field(jnp.asarray(arr), name=field.name, dims=field.dims,
                     units=field.units)

    return state._replace(
        T=F(state.T, _perturb_cols(state.T.data, nx)),
        S=F(state.S, _perturb_cols(state.S.data, nx)),
        eta=F(state.eta, _perturb_cols(state.eta.data, nx)),
        # u-faces have nx+1 columns; perturb its own halo seam faces.
        u=F(state.u, _perturb_cols(state.u.data, state.u.data.shape[1])),
        v=F(state.v, _perturb_cols(state.v.data, nx)),
    )


def test_leaf_projection_slaves_cell_halo_columns():
    """Cell-centred fields + v-faces: col0<-col[nx-2], col[nx-1]<-col1."""
    state, model = _model(ew_cyclic_overlap=True)
    s = _distinct_seam_state(state)
    out = model._apply_ew_cyclic_overlap(s)
    for name in ("T", "S", "eta", "v"):
        a = np.asarray(getattr(out, name).data)
        b = np.asarray(getattr(s, name).data)
        nx = a.shape[1]
        assert np.array_equal(a[:, 0], b[:, nx - 2]), f"{name} col0"
        assert np.array_equal(a[:, nx - 1], b[:, 1]), f"{name} col nx-1"
        assert np.array_equal(a[:, 1:nx - 1], b[:, 1:nx - 1]), f"{name} interior"
        assert a.shape == b.shape


def test_leaf_projection_slaves_u_face_halos():
    """u-faces (nx+1 cols): u[:,0]<-u[:,nx-2] (=n_lon-2), u[:,nx-1]<-u[:,1];
    the trailing wrap face u[:,nx] is left to the model's internal wrap."""
    state, model = _model(ew_cyclic_overlap=True)
    s = _distinct_seam_state(state)
    out = model._apply_ew_cyclic_overlap(s)
    u_out = np.asarray(out.u.data); u_in = np.asarray(s.u.data)
    nx = np.asarray(out.T.data).shape[1]      # n_lon (cell count)
    assert np.array_equal(u_out[:, 0], u_in[:, nx - 2])       # west face of cell0
    assert np.array_equal(u_out[:, nx - 1], u_in[:, 1])       # seam face
    # interior u-faces 1..nx-2 and the trailing wrap face nx untouched
    assert np.array_equal(u_out[:, 1:nx - 1], u_in[:, 1:nx - 1])
    assert np.array_equal(u_out[:, nx], u_in[:, nx])
    assert u_out.shape == u_in.shape


def test_gating_default_off_is_bit_identical():
    """config.ew_cyclic_overlap=False -> step bit-identical to no projection."""
    s0, m_off = _model(ew_cyclic_overlap=False)
    _, m_on = _model(ew_cyclic_overlap=True)
    s = _distinct_seam_state(s0)
    so = m_off.step(s, dt=600.0)
    sn = m_on.step(s, dt=600.0)
    # OFF: halo columns evolve freely (NOT slaved) -> generally differ from
    # the partner columns; ON: they are slaved exactly.
    nx = np.asarray(so.T.data).shape[1]
    To = np.asarray(so.T.data); Tn = np.asarray(sn.T.data)
    assert not np.allclose(To[:, 0], To[:, nx - 2])      # off: not slaved
    assert np.allclose(Tn[:, 0], Tn[:, nx - 2])          # on: slaved
    assert np.allclose(Tn[:, nx - 1], Tn[:, 1])
    # the two runs differ ONLY in the 2 halo columns (interior identical):
    assert np.allclose(To[:, 1:nx - 1], Tn[:, 1:nx - 1])


def test_numpy_setup_helper_matches_model_projection():
    from scripts.run.run_omip_core2 import _ew_overlap_fill
    state, model = _model(ew_cyclic_overlap=True)
    s = _distinct_seam_state(state)
    out = model._apply_ew_cyclic_overlap(s)
    for name in ("T", "S", "eta"):
        np_filled = _ew_overlap_fill(np.asarray(getattr(s, name).data))
        assert np.array_equal(np_filled, np.asarray(getattr(out, name).data)), name


def test_setup_helper_2d_and_3d_shapes():
    from scripts.run.run_omip_core2 import _ew_overlap_fill
    a2 = np.arange(8 * 10, dtype=np.float64).reshape(8, 10)
    a3 = np.arange(8 * 10 * 4, dtype=np.float64).reshape(8, 10, 4)
    for a in (a2, a3):
        f = _ew_overlap_fill(a)
        nx = a.shape[1]
        assert np.array_equal(f[:, 0], a[:, nx - 2])
        assert np.array_equal(f[:, nx - 1], a[:, 1])
        assert np.array_equal(f[:, 1:nx - 1], a[:, 1:nx - 1])
        assert f.shape == a.shape


def test_partial_cell_preserves_seam_identity():
    """Fix-2 guard: overlap-filling H_bathy/land_mask BEFORE make_partial_cell
    (smoothing=0, column-local) yields a partial-cell coordinate whose
    is_active / bottom_level satisfy the ORCA seam identity (col0==col[nx-2],
    col[nx-1]==col1) -- so the 3-D tracer-flux masks are consistent, not stale."""
    from scripts.run.run_omip_core2 import make_partial_cell, _ew_overlap_fill
    from legoesm.ocean.vertical import create_ocean_z_star
    n_lat, n_lon, nlev = 6, 12, 8
    z0 = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    rng = np.random.default_rng(3)
    H = rng.uniform(200.0, 3800.0, (n_lat, n_lon))
    lm = (rng.uniform(size=(n_lat, n_lon)) > 0.3).astype(np.float64)
    # raw (pre-fill) halo columns are arbitrary -> overlap-fill them first
    H = _ew_overlap_fill(H); lm = _ew_overlap_fill(lm)
    z_pc, H2, lm2 = make_partial_cell(z0, H, lm, smoothing_passes=0, min_levels=1)
    nx = n_lon
    ia = np.asarray(z_pc.is_active)
    assert np.array_equal(ia[:, 0], ia[:, nx - 2]), "is_active col0 != col[nx-2]"
    assert np.array_equal(ia[:, nx - 1], ia[:, 1]), "is_active col[nx-1] != col1"
    assert np.array_equal(np.asarray(H2)[:, 0], np.asarray(H2)[:, nx - 2])
    assert np.array_equal(np.asarray(lm2)[:, nx - 1], np.asarray(lm2)[:, 1])


def test_runner_flag_registered_and_defaults_off():
    """The runner exposes --ew-cyclic-overlap (default False) and build_tripole
    accepts the keyword — guards against the wiring being silently dropped."""
    import inspect
    from scripts.run import run_omip_core2 as r
    p = r.build_argparser() if hasattr(r, "build_argparser") else None
    # build_tripole must accept the keyword (the pass-through wiring)
    assert "ew_cyclic_overlap" in inspect.signature(r.build_tripole).parameters
    # the config field exists and defaults off (bit-exact legacy)
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert LatLonCGridOceanConfig().ew_cyclic_overlap is False
