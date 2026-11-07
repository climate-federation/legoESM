"""NEMO rn_shlat lateral momentum BC (``lateral_side_bc="nemo_shlat"``) vs NEMO ORCA1.

NEMO puts its lateral boundary condition in the VALUE of the f-mask: where the
free-slip f-mask is 0, ``fmask = rn_shlat * MIN(1, MAX(adjacent u/v masks))``
(dommsk.F90:209-215); ORCA1 runs rn_shlat = 2 (no slip).  legoESM's "no_slip"
is MITgcm's scalar side drag instead (tier-3 VIS finding V4).

Oracle quantity (campaign vote C6): the f-mask, compared exactly with NEMO
5.0.1 ORCA1's recorded fmask (values 0/1/2) on the same two 20x20x75 windows as
test_vis_level_masks_orca1_oracle (fixtures/orca1_t3/vis_level_masks.*).
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks,
    compute_vertex_mask,
    nemo_shlat_vertex_mask,
    per_level_viscosity_masks,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    _bc_horizontal_viscosity,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from pathlib import Path  # noqa: E402

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "vis_level_masks.npz"


def _sw(a):            # NEMO (i, j, k) -> port (lat=j, lon=i, k)
    return np.swapaxes(np.asarray(a, np.float64), 0, 1)


def _geo(n):
    return ensure_geometry(create_latlon_grid(n, n))


def _win(w):
    d = np.load(FIX)
    return _sw(d[f"w{w}_tmask"]), _sw(d[f"w{w}_fmask"])


@pytest.mark.parametrize("w", [0, 1])
def test_shlat_fmask_equals_nemo_fmask(w):
    t, fm = _win(w); n = t.shape[0]
    _, um3, vm3, vx3 = per_level_viscosity_masks(
        jnp.asarray(t), _geo(n), None, None, None, None, jnp.float64)
    f = np.asarray(nemo_shlat_vertex_mask(vx3, um3, vm3, 2.0))
    np.testing.assert_array_equal(f[1:n, 1:n], fm[:n - 1, :n - 1])


@pytest.mark.parametrize("w", [0, 1])
def test_fixture_has_no_slip_points(w):
    """Control: NEMO's f-mask has value-2 coastal points here, so the test
    above checks the rn_shlat rule, not just the free-slip product."""
    assert int((_win(w)[1] == 2).sum()) > 0


def test_shlat_zero_is_free_slip():
    t, _ = _win(0); n = t.shape[0]
    _, um3, vm3, vx3 = per_level_viscosity_masks(
        jnp.asarray(t), _geo(n), None, None, None, None, jnp.float64)
    np.testing.assert_array_equal(
        np.asarray(nemo_shlat_vertex_mask(vx3, um3, vm3, 0.0)), np.asarray(vx3))


def _call(cfg, t, seed=11):
    n, nk = t.shape[0], t.shape[-1]; geo = _geo(n)
    rng = np.random.default_rng(seed)
    u = jnp.asarray(rng.standard_normal((n, n + 1, nk)))
    v = jnp.asarray(rng.standard_normal((n + 1, n, nk)))
    mask = jnp.asarray(t[..., 0]); um, vm = compute_face_masks(mask, geo)
    vx = compute_vertex_mask(mask, grid=geo)
    zc = types.SimpleNamespace(is_active=jnp.asarray(t > 0))
    o = _bc_horizontal_viscosity(jnp.zeros_like(u), jnp.zeros_like(v), u, v, geo,
                                 mask, um, vm, cfg, zc, None, 150.0, vertex_mask=vx)
    return np.asarray(o[0]), np.asarray(o[1]), (geo, u, v, zc, mask, um, vm, vx)


def test_nemo_shlat_tendency_uses_the_shlat_vertex_mask():
    """nemo_shlat == the per_level operator fed the rn_shlat vertex mask level
    by level; no MITgcm side drag is added; and it differs from free slip."""
    t, _ = _win(1)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, lateral_viscosity_mask_levels="per_level",
        lateral_side_bc="nemo_shlat", lateral_shlat=2.0)
    du, dv, (geo, u, v, zc, mask, um, vm, vx) = _call(cfg, t)
    cell, um3, vm3, vx3 = per_level_viscosity_masks(
        zc.is_active, geo, mask, um, vm, vx, jnp.float64)
    f3 = nemo_shlat_vertex_mask(vx3, um3, vm3, 2.0)
    base = cfg._replace(lateral_viscosity_mask_levels="surface", lateral_side_bc="free_slip")
    ru, rv = [], []
    for k in range(t.shape[-1]):
        o = _bc_horizontal_viscosity(
            jnp.zeros_like(u[..., k:k + 1]), jnp.zeros_like(v[..., k:k + 1]),
            u[..., k:k + 1], v[..., k:k + 1], geo, cell[..., k], um3[..., k], vm3[..., k],
            base, types.SimpleNamespace(), None, 150.0, vertex_mask=f3[..., k])
        ru.append(np.asarray(o[0])[..., 0]); rv.append(np.asarray(o[1])[..., 0])
    np.testing.assert_allclose(du, np.stack(ru, -1), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(dv, np.stack(rv, -1), rtol=1e-12, atol=0.0)
    fu, fv, _ = _call(cfg._replace(lateral_side_bc="free_slip"), t)
    assert not np.array_equal(du, fu) and not np.array_equal(dv, fv)


def test_nemo_shlat_requires_per_level_and_nonnegative_shlat():
    t, _ = _win(0)
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1.0e4, lateral_side_bc="nemo_shlat",
                                           lateral_viscosity_mask_levels="surface")
    with pytest.raises(ValueError, match="per_level"):
        _call(cfg, t)
    with pytest.raises(ValueError, match="lateral_shlat"):
        _call(cfg._replace(lateral_viscosity_mask_levels="per_level", lateral_shlat=-1.0), t)


def test_fold_row_is_refilled_from_its_partners_after_shlat():
    """NEMO calls lbc_lnk(fmask, 'F') after setting rn_shlat: on a tripolar
    fold the last vertex row is the fold image of the row below, rn_shlat
    values included (codex r1)."""
    t, _ = _win(1); n = t.shape[0]
    perm = jnp.asarray((n - 1 - np.arange(n)) % n)
    fold = types.SimpleNamespace(is_active=True, fold_j=n - 1, cap_j=n - 1,
                                 perm_T=perm, perm_v=perm, perm_f=None)
    grid = types.SimpleNamespace(fold=fold)
    _, um3, vm3, vx3 = per_level_viscosity_masks(
        jnp.asarray(t), _geo(n), None, None, None, None, jnp.float64)
    f = np.asarray(nemo_shlat_vertex_mask(vx3, um3, vm3, 2.0, grid=grid))
    row = f[-2][:n][np.asarray(perm)]
    np.testing.assert_array_equal(f[-1][:n], row)
    assert np.array_equal(f[-1][n], f[-1][0])          # wrap column
    assert (f[-2] == 2).any()                          # the fold copies shlat values
