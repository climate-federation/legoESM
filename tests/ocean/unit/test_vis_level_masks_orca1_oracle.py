"""Per-level land masks of the vector-Laplacian lateral viscosity vs NEMO ORCA1.

NEMO's dyn_ldf_lev reads 3-D umask/vmask/fmask built level by level from tmask
(dommsk.F90:148-153).  The "vector_laplacian" path applied the 2-D surface
masks at every level (tier-3 VIS finding V3: 0.28/0.22 RMS of the tendency).
``lateral_viscosity_mask_levels="per_level"`` builds them per level.

Oracle quantity (campaign vote C6): the masks themselves, compared exactly with
NEMO 5.0.1 ORCA1's recorded umask/vmask/fe3mask (the free-slip f-mask; NEMO's
rn_shlat fmask is a separate boundary-condition choice).  Fixture: two 20x20
windows, all 75 levels, picked for topographic steps; provenance in
fixtures/orca1_t3/vis_level_masks.json.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks,
    compute_vertex_mask,
    per_level_viscosity_masks,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    _bc_horizontal_viscosity,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "vis_level_masks.npz"
PROV = FIX.with_suffix(".json")


def _windows():
    d = np.load(FIX)
    n_w = len(json.loads(PROV.read_text())["windows"])
    return [{k: d[f"w{w}_{k}"] for k in ("tmask", "umask", "vmask", "fe3mask")}
            for w in range(n_w)]


def _sw(a):            # NEMO (i, j, k) -> port (lat=j, lon=i, k)
    return np.swapaxes(np.asarray(a, np.float64), 0, 1)


def _geo(n):
    return ensure_geometry(create_latlon_grid(n, n))


@pytest.mark.parametrize("w", [0, 1])
def test_per_level_masks_equal_nemo_umask_vmask_fmask(w):
    """Every interior face / vertex at every level equals NEMO's mask exactly."""
    win = _windows()[w]
    t = _sw(win["tmask"]); n = t.shape[0]
    _, um3, vm3, vx3 = per_level_viscosity_masks(
        jnp.asarray(t), _geo(n), None, None, None, None, jnp.float64)
    # port u-face c (1..n-1) is NEMO u(c-1): east face of cell c-1
    np.testing.assert_array_equal(np.asarray(um3)[:, 1:n], _sw(win["umask"])[:, :n - 1])
    np.testing.assert_array_equal(np.asarray(vm3)[1:n], _sw(win["vmask"])[:n - 1])
    np.testing.assert_array_equal(np.asarray(vx3)[1:n, 1:n], _sw(win["fe3mask"])[:n - 1, :n - 1])


@pytest.mark.parametrize("w", [0, 1])
def test_fixture_has_steps_the_surface_masks_miss(w):
    """Control: the surface masks broadcast to depth differ from NEMO's 3-D
    masks here, so the test above can fail."""
    win = _windows()[w]
    for k in ("umask", "vmask", "fe3mask"):
        m = win[k]
        assert int(((m != m[..., :1]) & (m[..., :1] > 0)).sum()) > 0, k


def _call(cfg, geo, t, u, v, z_coord):
    mask = jnp.asarray(t[..., 0])
    um, vm = compute_face_masks(mask, geo)
    vx = compute_vertex_mask(mask, grid=geo)
    out = _bc_horizontal_viscosity(
        jnp.zeros_like(u), jnp.zeros_like(v), u, v, geo, mask, um, vm, cfg,
        z_coord, None, 150.0, vertex_mask=vx)
    return np.asarray(out[0]), np.asarray(out[1]), (mask, um, vm, vx)


def _fields(n, nk, seed):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.standard_normal((n, n + 1, nk))),
            jnp.asarray(rng.standard_normal((n + 1, n, nk))))


@pytest.mark.parametrize("w", [0, 1])
def test_per_level_tendency_is_the_level_by_level_operator(w):
    """per_level == the same operator called one level at a time with that
    level's masks (fed as 2-D masks), and differs from the surface default."""
    t = _sw(_windows()[w]["tmask"]); n, nk = t.shape[0], t.shape[-1]
    geo = _geo(n)
    u, v = _fields(n, nk, 7 + w)
    zc = types.SimpleNamespace(is_active=jnp.asarray(t > 0))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, lateral_viscosity_mask_levels="per_level")
    du, dv, surf = _call(cfg, geo, t, u, v, zc)
    cell, um3, vm3, vx3 = per_level_viscosity_masks(
        zc.is_active, geo, *surf, jnp.float64)
    base = cfg._replace(lateral_viscosity_mask_levels="surface")
    ref_u, ref_v = [], []
    for k in range(nk):
        o = _bc_horizontal_viscosity(
            jnp.zeros_like(u[..., k:k + 1]), jnp.zeros_like(v[..., k:k + 1]),
            u[..., k:k + 1], v[..., k:k + 1], geo, cell[..., k], um3[..., k],
            vm3[..., k], base, types.SimpleNamespace(), None, 150.0,
            vertex_mask=vx3[..., k])
        ref_u.append(np.asarray(o[0])[..., 0]); ref_v.append(np.asarray(o[1])[..., 0])
    np.testing.assert_allclose(du, np.stack(ref_u, -1), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(dv, np.stack(ref_v, -1), rtol=1e-12, atol=0.0)
    su, sv, _ = _call(base, geo, t, u, v, zc)
    assert not np.array_equal(du, su) and not np.array_equal(dv, sv)


def test_surface_default_is_unchanged_and_no_is_active_falls_back():
    t = _sw(_windows()[0]["tmask"]); n, nk = t.shape[0], t.shape[-1]
    geo = _geo(n); u, v = _fields(n, nk, 3)
    zc = types.SimpleNamespace(is_active=jnp.asarray(t > 0))
    dflt = LatLonCGridOceanConfig.from_flat(A_h=1.0e4)
    assert dflt.lateral_viscosity_mask_levels == "surface"
    a = _call(dflt, geo, t, u, v, zc)
    b = _call(dflt._replace(lateral_viscosity_mask_levels="per_level"), geo, t, u, v,
              types.SimpleNamespace())
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])


def test_unknown_or_unsupported_selection_raises():
    t = _sw(_windows()[0]["tmask"]); n, nk = t.shape[0], t.shape[-1]
    geo = _geo(n); u, v = _fields(n, nk, 5)
    zc = types.SimpleNamespace(is_active=jnp.asarray(t > 0))
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1.0e4)
    with pytest.raises(ValueError, match="lateral_viscosity_mask_levels"):
        _call(cfg._replace(lateral_viscosity_mask_levels="3d"), geo, t, u, v, zc)
    with pytest.raises(ValueError, match="per_level"):
        _call(cfg._replace(lateral_viscosity_mask_levels="per_level",
                           lateral_viscosity_operator="flux_divergence"), geo, t, u, v, zc)


def test_3d_vertex_mask_equals_the_per_level_2d_mask():
    """compute_vertex_mask on a (n_lat, n_lon, nlev) mask (one pad / halo
    exchange for all levels) == the 2-D mask of each level."""
    t = _sw(_windows()[1]["tmask"]); geo = _geo(t.shape[0])
    v3 = np.asarray(compute_vertex_mask(jnp.asarray(t), grid=geo))
    for k in range(t.shape[-1]):
        np.testing.assert_array_equal(
            v3[..., k], np.asarray(compute_vertex_mask(jnp.asarray(t[..., k]), grid=geo)))
