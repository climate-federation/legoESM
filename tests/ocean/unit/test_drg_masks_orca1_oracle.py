"""NEMO 5.0.1 ORCA1 oracle for the bottom-drag rate masks (zdf_drg_nonlin, np_non_lin).

Fixture ``fixtures/orca1_t3/drg_masks.npz`` (+ ``.json`` provenance): two 16x16
windows of NEMO's recorded zdf_drg inputs (uu/vv at Kmm, tmask, mbkt) and output
(``rCdU_bot``) at kt 17281 of the ORCA1 day-30 restart, two of the 32 MPI tiles.

NEMO: ``rCdU_bot = -rCd0_bot*SQRT(0.25*(zut**2+zvt**2)+rn_ke0)`` with
``rCd0_bot = rn_Cd0*ssmask`` (zdfdrg.F90:186-190, :465, :497) and ``uu`` umask'ed
at every level, so (a) land columns carry 0 and (b) the rate does not depend on
velocities stored on faces below the seafloor.  legoESM's rate
``r = -rCdU_bot >= 0`` (:func:`nemo_bottom_drag_rate_faces`).

Scored on the window interior (outer ring dropped: its face masks need a
neighbour outside the window).  Tolerance: rel 1e-12 (NEMO 20-seed spread
8.8e-16, campaign vote C2).
"""

from __future__ import annotations

import json
import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_bottom_drag_rate_faces
from legoesm.ocean.state import DynBottomDragConfig
from legoesm.ocean.vertical import OceanPartialCellCoordinate

jax.config.update("jax_enable_x64", True)

_FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t3" / "drg_masks"
_DATA = np.load(str(_FIX) + ".npz")
_PROV = json.loads(pathlib.Path(str(_FIX) + ".json").read_text())
_TAGS = [f"r{w['rank']}" for w in _PROV["windows"]]
_TOL = 1e-12


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(before)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)))


def _window(tag):
    w = next(x for x in _PROV["windows"] if f"r{x['rank']}" == tag)
    sc = w["scalars"]
    u = jnp.asarray(_DATA[f"{tag}_u"])
    v = jnp.asarray(_DATA[f"{tag}_v"])
    act = _DATA[f"{tag}_is_active"].astype(bool)
    bl = _DATA[f"{tag}_bottom_level"]
    nj, ni, nk = act.shape
    nan1 = jnp.full((nk,), jnp.nan)
    nan3 = jnp.full((nj, ni, nk), jnp.nan)
    zc = OceanPartialCellCoordinate(
        n_levels=nk, H_max=float("nan"), z_full_ref=nan1,
        z_half_ref=jnp.full((nk + 1,), jnp.nan), dz_ref=nan1, dz_half_ref=nan1,
        h_partial=nan3, bottom_level=jnp.asarray(bl, jnp.int32),
        is_active=jnp.asarray(act))
    cfg = types.SimpleNamespace(bottom_drag=DynBottomDragConfig(
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=sc["cd0"],
        bottom_drag_ke0=sc["ke0"], bottom_drag_cdmax=sc["cdmax"],
        bottom_drag_z0=sc["z0"]))
    # h_k is NaN: the quadratic law must not read it (non-finite output fails).
    f = jax.jit(lambda U, V: nemo_bottom_drag_rate_faces(
        U, V, nan3, zc, cfg, None, return_cell_rate=True))
    return u, v, act, bl, _DATA[f"{tag}_nemo_r"], f


def _dry_faces(act):
    """3-D dry u/v faces (port layout), from tmask alone, numpy."""
    a = act.astype(np.float64)
    um = np.zeros((a.shape[0], a.shape[1] + 1, a.shape[2]))
    um[:, 1:-1] = a[:, 1:] * a[:, :-1]
    vm = np.zeros((a.shape[0] + 1, a.shape[1], a.shape[2]))
    vm[1:-1] = a[1:] * a[:-1]
    return um == 0, vm == 0


@pytest.mark.parametrize("tag", _TAGS)
def test_cell_rate_matches_nemo_including_land(tag):
    u, v, act, bl, nemo, f = _window(tag)
    r_t = np.asarray(f(u, v)[4])
    assert np.all(np.isfinite(r_t))
    inner = (slice(1, -1), slice(1, -1))
    land = bl[inner] < 0
    assert land.any() and (~land).any()
    assert np.all(r_t[inner][land] == 0.0), "land rate must be 0 (rCd0_bot = rn_Cd0*ssmask)"
    assert _rel(r_t[inner], nemo[inner]) <= _TOL


@pytest.mark.parametrize("tag", _TAGS)
def test_face_coefficients_match_nemo_dynzdf_average(tag):
    u, v, act, bl, nemo, f = _window(tag)
    r_u, r_v = (np.asarray(x) for x in f(u, v)[:2])
    # dynzdf zCdu = 0.5*(rCdU(ji+1)+rCdU(ji)): port u-face i couples cells i-1, i.
    nu = 0.5 * (nemo[1:-1, 1:-2] + nemo[1:-1, 2:-1])
    nv = 0.5 * (nemo[1:-2, 1:-1] + nemo[2:-1, 1:-1])
    assert _rel(r_u[1:-1, 2:-2], nu) <= _TOL
    assert _rel(r_v[2:-2, 1:-1], nv) <= _TOL


@pytest.mark.parametrize("tag", _TAGS)
def test_rate_ignores_velocity_below_the_seafloor(tag):
    """NEMO's rate cannot see sub-seafloor faces (uu umask'ed): fill them with
    seeded noise and require the NEMO value unchanged (vote V1)."""
    u, v, act, bl, nemo, f = _window(tag)
    du, dv = _dry_faces(act)
    rng = np.random.default_rng(20261103)
    u_n = np.where(du, rng.uniform(-0.1, 0.1, du.shape), np.asarray(u))
    v_n = np.where(dv, rng.uniform(-0.1, 0.1, dv.shape), np.asarray(v))
    r_t = np.asarray(f(jnp.asarray(u_n), jnp.asarray(v_n))[4])
    inner = (slice(1, -1), slice(1, -1))
    assert _rel(r_t[inner], nemo[inner]) <= _TOL


def test_rate_gradient_finite_and_blind_to_dry_faces():
    tag = _TAGS[0]
    u, v, act, bl, nemo, f = _window(tag)
    du, _ = _dry_faces(act)
    g = np.asarray(jax.grad(lambda U: jnp.sum(f(U, v)[4][1:-1, 1:-1]))(u))
    assert np.all(np.isfinite(g))
    assert np.all(g[du] == 0.0)
    # one wet bottom face of an interior wet cell: central FD ratio ~ 1
    un = np.asarray(u)
    j, i = next((jj, ii) for jj, ii in np.argwhere(bl[1:-1, 1:-1] >= 0) + 1
                if act[jj, ii + 1, bl[jj, ii]] and un[jj, ii + 1, bl[jj, ii]] != 0.0)
    k = int(bl[j, i])
    eps = 1e-6
    e = np.zeros(u.shape)
    e[j, i + 1, k] = eps
    fd = (float(jnp.sum(f(u + e, v)[4][1:-1, 1:-1]))
          - float(jnp.sum(f(u - e, v)[4][1:-1, 1:-1]))) / (2 * eps)
    assert g[j, i + 1, k] != 0.0
    assert abs(fd / g[j, i + 1, k] - 1.0) < 1e-6


@pytest.mark.parametrize("fold_active", [False, True])
def test_north_fold_v_row_passes_through_only_on_local_fold(fold_active):
    """The helper closes the last v row; on the rank holding the fold seam
    (fold_j >= 0) that row is left unmasked (pre-fix behaviour, the fold
    vmask can be wet); a non-local fold (fold_j = -1) keeps the 3-D mask."""
    tag = _TAGS[0]
    u, v, act, bl, nemo, f = _window(tag)
    nj, ni, nk = act.shape
    nan1 = jnp.full((nk,), jnp.nan)
    nan3 = jnp.full((nj, ni, nk), jnp.nan)
    zc = OceanPartialCellCoordinate(
        n_levels=nk, H_max=float("nan"), z_full_ref=nan1,
        z_half_ref=jnp.full((nk + 1,), jnp.nan), dz_ref=nan1, dz_half_ref=nan1,
        h_partial=nan3, bottom_level=jnp.asarray(bl, jnp.int32),
        is_active=jnp.asarray(act))
    sc = next(x for x in _PROV["windows"] if f"r{x['rank']}" == tag)["scalars"]
    cfg = types.SimpleNamespace(bottom_drag=DynBottomDragConfig(
        bottom_drag_scheme="nemo_quadratic", bottom_drag_cd0=sc["cd0"],
        bottom_drag_ke0=sc["ke0"], bottom_drag_cdmax=sc["cdmax"],
        bottom_drag_z0=sc["z0"]))
    grid = types.SimpleNamespace(fold=types.SimpleNamespace(
        is_active=True, fold_j=0 if fold_active else -1))
    wet_top = np.flatnonzero(bl[-1] >= 0)
    assert wet_top.size
    v2 = np.asarray(v).copy()
    v2[-1, wet_top, bl[-1, wet_top]] += 0.05
    r0, r1 = (np.asarray(nemo_bottom_drag_rate_faces(
        u, jnp.asarray(x), nan3, zc, cfg, grid, return_cell_rate=True)[4][-1])
        for x in (v, v2))
    if fold_active:
        assert np.all(r1[wet_top] != r0[wet_top])
    else:
        assert np.array_equal(r1, r0)
