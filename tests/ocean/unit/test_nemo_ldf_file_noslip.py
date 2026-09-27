"""NEMO nn_ahm_ijk_t=-30 viscosity path: file coefficients + rn_shlat fmask.

* ``nemo_fmask_shlat_3d`` reproduces dommsk.F90:207-210 on hand-checkable
  land patterns (single land cell, diagonal pair);
* full-shape (3-D) coefficients reduce exactly to the 1-D-coefficient path;
* the ``h_vtx`` override is what keeps no-slip alive in the e3-weighted
  operator (the min-rule gives e3f = 0 at the coast, silently free-slip);
* the operator dissipates kinetic energy with no-slip walls;
* ``_bc_horizontal_viscosity`` routes coordinate-carried NEMO fields to the
  operator and refuses a nonzero A_h next to them.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    nemo_fmask_shlat_3d,
    nemo_ldf_lap_viscosity_cgrid,
    nemo_ldf_lap_viscosity_e3_cgrid,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_horizontal_viscosity
from legoesm.ocean.state import LatLonCGridOceanConfig

NLAT, NLON = 24, 32


def _geo():
    return ensure_geometry(create_latlon_grid(NLAT, NLON))


def _active(land_cells=(), nk=2):
    a = np.ones((NLAT, NLON, nk), bool)
    a[0] = a[-1] = False                     # keep the polar rows out of play
    for j, i in land_cells:
        a[j, i] = False
    return jnp.asarray(a)


def test_fmask_single_land_cell_corners_take_rn_shlat():
    geo = _geo()
    j, i = 12, 16
    f2 = np.asarray(nemo_fmask_shlat_3d(_active([(j, i)]), geo, 2.0))[..., 0]
    f0 = np.asarray(nemo_fmask_shlat_3d(_active([(j, i)]), geo, 0.0))[..., 0]
    corners = [(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)]   # vertex = SW corner of T
    for c in corners:
        assert f2[c] == 2.0 and f0[c] == 0.0, c
    ring = f2[6:18, 10:22].copy()
    for (a, b) in corners:
        ring[a - 6, b - 10] = 1.0
    assert np.all(ring == 1.0)


def test_fmask_diagonal_pair_shared_corner_is_zero():
    """Two land cells touching only at a corner: no velocity face at the shared
    F point is wet, so NEMO's MIN(1, MAX(umask.., vmask..)) is 0, not rn_shlat."""
    geo = _geo()
    f2 = np.asarray(nemo_fmask_shlat_3d(_active([(12, 16), (13, 17)]), geo, 2.0))[..., 0]
    assert f2[13, 17] == 0.0          # shared corner
    assert f2[12, 16] == 2.0 and f2[14, 18] == 2.0


def _uv(rng, nk=2):
    return (jnp.asarray(rng.standard_normal((NLAT, NLON + 1, nk))),
            jnp.asarray(rng.standard_normal((NLAT + 1, NLON, nk))))


def _masks(act):
    um, vm = compute_face_masks_3d(act, None)
    return act.astype(float), um.astype(float), vm.astype(float)


def test_full_shape_coefficients_reduce_to_1d_path():
    geo = _geo()
    act = _active([(12, 16)])
    cm, um, vm = _masks(act)
    u, v = _uv(np.random.default_rng(0))
    A = 1.3e4
    t1 = jnp.full((NLAT,), A); f1 = jnp.full((NLAT + 1,), A)
    t3 = jnp.full((NLAT, NLON, 2), A); f3 = jnp.full((NLAT + 1, NLON + 1, 2), A)
    kw = dict(mask=cm[..., 0], u_mask=um, v_mask=vm)
    for op, extra in ((nemo_ldf_lap_viscosity_cgrid, ()),
                      (nemo_ldf_lap_viscosity_e3_cgrid, (jnp.full((NLAT, NLON, 2), 50.0),))):
        a = op(u, v, geo, t1, f1, *extra, **kw)
        b = op(u, v, geo, t3, f3, *extra, **kw)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _noslip_call(act, h, shlat, h_vtx, rng_seed=1):
    geo = _geo()
    cm, um, vm = _masks(act)
    fm = nemo_fmask_shlat_3d(act, geo, shlat)
    A = 1.0e4
    ahmt = A * cm
    ahmf = A * fm
    u, v = _uv(np.random.default_rng(rng_seed))
    u, v = u * um, v * vm
    return geo, (u, v, um, vm), nemo_ldf_lap_viscosity_e3_cgrid(
        u, v, geo, ahmt, ahmf, h, mask=cm[..., 0], u_mask=um, v_mask=vm,
        vertex_mask=(ahmf > 0).astype(float), h_vtx=h_vtx)


def test_h_vtx_override_keeps_noslip_alive():
    act = _active([(12, 16), (12, 17), (13, 16)])
    h = jnp.where(act, 50.0, 0.0)
    h_fill = jnp.full((NLAT + 1, NLON + 1, 2), 50.0)
    _, _, bug2 = _noslip_call(act, h, 2.0, None)        # min-rule: e3f=0 at coast
    _, _, free = _noslip_call(act, h, 0.0, None)
    _, _, ok2 = _noslip_call(act, h, 2.0, h_fill)
    np.testing.assert_allclose(np.asarray(bug2[0]), np.asarray(free[0]), atol=1e-12)
    scale = np.max(np.abs(np.asarray(free[0])))
    assert np.max(np.abs(np.asarray(ok2[0]) - np.asarray(free[0]))) > 0.01 * scale
    # uniform thickness + filled e3f: e3 operator == unweighted operator
    geo = _geo()
    cm, um, vm = _masks(act)
    fm = nemo_fmask_shlat_3d(act, geo, 2.0)
    u, v = _uv(np.random.default_rng(1)); u, v = u * um, v * vm
    ref = nemo_ldf_lap_viscosity_cgrid(u, v, geo, 1.0e4 * cm, 1.0e4 * fm,
                                       mask=cm[..., 0], u_mask=um, v_mask=vm,
                                       vertex_mask=(fm > 0).astype(float))
    np.testing.assert_allclose(np.asarray(ok2[0]), np.asarray(ref[0]), rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("shlat", [0.0, 2.0])
def test_operator_dissipates_kinetic_energy(shlat):
    act = _active([(10, 10), (10, 11), (11, 10), (15, 20), (16, 21)])
    h = jnp.where(act, 50.0, 0.0)
    geo, (u, v, um, vm), (tu, tv) = _noslip_call(
        act, h, shlat, jnp.full((NLAT + 1, NLON + 1, 2), 50.0), rng_seed=7)
    au = np.asarray(geo.dx_u * geo.dy_u)[..., None][:, :-1]   # drop duplicate seam col
    av = np.asarray(geo.dx_v * geo.dy_v)[..., None]
    dE = (np.sum(au * 50.0 * np.asarray(u)[:, :-1] * np.asarray(tu)[:, :-1])
          + np.sum(av * 50.0 * np.asarray(v) * np.asarray(tv)))
    assert dE < 0.0


def test_bc_horizontal_viscosity_routes_file_fields_and_refuses_A_h():
    geo = _geo()
    act = _active([(12, 16)])
    cm, um, vm = _masks(act)
    fm = nemo_fmask_shlat_3d(act, geo, 2.0)
    h = jnp.where(act, 50.0, 0.0)
    e3f = jnp.full((NLAT + 1, NLON + 1, 2), 50.0)
    zc = SimpleNamespace(is_active=act, nemo_ahmt_3d=1.0e4 * cm,
                         nemo_ahmf_3d=1.0e4 * fm, nemo_e3f_0=e3f)
    u, v = _uv(np.random.default_rng(3)); u, v = u * um, v * vm
    z0u, z0v = jnp.zeros_like(u), jnp.zeros_like(v)
    mask2 = cm[..., 0]; um2 = um[..., 0]; vm2 = vm[..., 0]
    cfg = LatLonCGridOceanConfig.from_flat(
        lateral_viscosity_operator="nemo_div_curl",
        lateral_viscosity_e3_weighting="nemo_e3", A_h=0.0, C_smag_lap=0.0)
    out = _bc_horizontal_viscosity(z0u, z0v, u, v, geo, mask2, um2, vm2, cfg,
                                   zc, None, 1.0, h_k=h)
    ref, _ = nemo_ldf_lap_viscosity_e3_cgrid(
        u, v, geo, zc.nemo_ahmt_3d, zc.nemo_ahmf_3d, h, mask=mask2,
        u_mask=um, v_mask=vm, vertex_mask=(zc.nemo_ahmf_3d > 0).astype(float),
        h_vtx=e3f)
    assert float(np.max(np.abs(np.asarray(ref)))) > 0.0
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref), rtol=1e-12, atol=1e-20)
    with pytest.raises(ValueError, match="A_h must be 0"):
        _bc_horizontal_viscosity(z0u, z0v, u, v, geo, mask2, um2, vm2,
                                 cfg.replace_flat(A_h=1.0e4), zc, None, 1.0, h_k=h)
