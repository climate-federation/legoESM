"""EXT-N1: 3-D (full-rank per-cell) sponge gamma.

The Veros north_atlantic setup restores T/S with a STATIC 3-D rate field
``rest_tscl(x, y, z)`` (north_atlantic.py:102/257-261; consumed as
``temp_source = maskT · rest_tscl · (t* − T)``, :346-356).  Verified in the
real data (transfer-matrix scoping): 15/436 sponge columns are z-VARYING and
401/436 are nonzero over only part of the column — a 2-D gamma cannot
represent it.

``apply_sponge_tracer_relaxation`` (ocean/dynamics/ocean_tendency_common.py)
now shape-dispatches on ``gamma.ndim``:

  * ``T.ndim - 1`` (horizontal) → ``expand_dims`` — the ORIGINAL code path,
    byte-identical ⇒ 2-D bit-identity;
  * ``T.ndim`` (full-rank per-cell) → used as-is;
  * anything else → ValueError (trace-time; ndim is static).

A full-rank gamma supports TRACER relaxation only: ``_bc_sponge_relaxation``
raises when combined with ``u_ref``/``v_ref`` (the momentum sponge
interpolates gamma to velocity faces, defined for the horizontal form only).

Run in fp64 on CPU for determinism.
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

from legoesm.ocean.dynamics.ocean_tendency_common import (
    apply_sponge_tracer_relaxation,
)
from legoesm.ocean.sponge import SpongeForcing

_N_LAT, _N_LON, _NLEV = 3, 4, 5


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.normal(10.0, 2.0, size=(_N_LAT, _N_LON, _NLEV)))
    S = jnp.asarray(rng.normal(35.0, 0.3, size=(_N_LAT, _N_LON, _NLEV)))
    T_ref = jnp.asarray(rng.normal(12.0, 2.0, size=(_N_LAT, _N_LON, _NLEV)))
    S_ref = jnp.asarray(rng.normal(34.5, 0.3, size=(_N_LAT, _N_LON, _NLEV)))
    return T, S, T_ref, S_ref


# ---------------------------------------------------------------------------
# Hand-computed values
# ---------------------------------------------------------------------------

def test_gamma_3d_hand_computed():
    """3-D gamma: dT = γ(x,y,z)·(T_ref − T) cell by cell, exactly."""
    T, S, T_ref, S_ref = _fields()
    rng = np.random.default_rng(1)
    g3 = jnp.asarray(rng.uniform(0.0, 1e-5, size=(_N_LAT, _N_LON, _NLEV)))
    sp = SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref)
    z = jnp.zeros_like(T)
    dT, dS = apply_sponge_tracer_relaxation(z, z, T, S, sp)
    np.testing.assert_array_equal(np.asarray(dT), np.asarray(g3 * (T_ref - T)))
    np.testing.assert_array_equal(np.asarray(dS), np.asarray(g3 * (S_ref - S)))


def test_gamma_3d_partial_column_and_z_varying():
    """A partial-column, z-varying gamma (the NA rest_tscl structure) relaxes
    only where γ ≠ 0 and with the per-level rate."""
    T, S, T_ref, S_ref = _fields()
    g3 = np.zeros((_N_LAT, _N_LON, _NLEV))
    g3[1, 2, 0] = 1.0e-5          # surface-only column
    g3[2, 1, :3] = [3e-6, 2e-6, 1e-6]   # z-varying partial column
    g3 = jnp.asarray(g3)
    sp = SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref)
    z = jnp.zeros_like(T)
    dT, _ = apply_sponge_tracer_relaxation(z, z, T, S, sp)
    dT = np.asarray(dT)
    # untouched outside the support
    assert dT[0].max() == 0.0 and abs(dT[1, 2, 1:]).max() == 0.0
    assert abs(dT[2, 1, 3:]).max() == 0.0
    # hand value on the z-varying column
    expect = np.asarray(g3[2, 1, :3] * (T_ref[2, 1, :3] - T[2, 1, :3]))
    np.testing.assert_array_equal(dT[2, 1, :3], expect)


# ---------------------------------------------------------------------------
# 2-D bit-identity (the original path must be byte-identical)
# ---------------------------------------------------------------------------

def test_gamma_2d_bit_identical_to_manual_expand_dims():
    T, S, T_ref, S_ref = _fields(2)
    rng = np.random.default_rng(3)
    g2 = jnp.asarray(rng.uniform(0.0, 1e-5, size=(_N_LAT, _N_LON)))
    sp = SpongeForcing(gamma=g2, T_ref=T_ref, S_ref=S_ref)
    z = jnp.zeros_like(T)
    dT, dS = apply_sponge_tracer_relaxation(z, z, T, S, sp)
    # The pre-EXT-N1 formula, verbatim.
    ref_T = jnp.expand_dims(g2, -1) * (T_ref - T)
    ref_S = jnp.expand_dims(g2, -1) * (S_ref - S)
    np.testing.assert_array_equal(np.asarray(dT), np.asarray(ref_T))
    np.testing.assert_array_equal(np.asarray(dS), np.asarray(ref_S))


def test_gamma_3d_constant_in_z_matches_2d():
    """A 3-D gamma that is constant along z is numerically identical to the
    broadcast 2-D gamma (same multiplications)."""
    T, S, T_ref, S_ref = _fields(4)
    rng = np.random.default_rng(5)
    g2 = jnp.asarray(rng.uniform(0.0, 1e-5, size=(_N_LAT, _N_LON)))
    g3 = jnp.broadcast_to(g2[..., None], T.shape)
    z = jnp.zeros_like(T)
    dT2, dS2 = apply_sponge_tracer_relaxation(
        z, z, T, S, SpongeForcing(gamma=g2, T_ref=T_ref, S_ref=S_ref))
    dT3, dS3 = apply_sponge_tracer_relaxation(
        z, z, T, S, SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref))
    np.testing.assert_array_equal(np.asarray(dT2), np.asarray(dT3))
    np.testing.assert_array_equal(np.asarray(dS2), np.asarray(dS3))


# ---------------------------------------------------------------------------
# Masking + MPAS shapes
# ---------------------------------------------------------------------------

def test_gamma_3d_with_horizontal_mask():
    T, S, T_ref, S_ref = _fields(6)
    rng = np.random.default_rng(7)
    g3 = jnp.asarray(rng.uniform(0.0, 1e-5, size=T.shape))
    mask = np.ones((_N_LAT, _N_LON)); mask[0, :] = 0.0
    mask = jnp.asarray(mask)
    z = jnp.zeros_like(T)
    dT, _ = apply_sponge_tracer_relaxation(
        z, z, T, S, SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref),
        mask=mask)
    dT = np.asarray(dT)
    assert abs(dT[0]).max() == 0.0
    np.testing.assert_array_equal(
        dT[1:], np.asarray((g3 * (T_ref - T)))[1:])


def test_mpas_shapes_1d_and_2d_gamma():
    """MPAS layout: T is (nCells, nlev); gamma (nCells,) broadcasts, gamma
    (nCells, nlev) is per-cell."""
    rng = np.random.default_rng(8)
    n_cells, nlev = 7, 4
    T = jnp.asarray(rng.normal(10.0, 2.0, size=(n_cells, nlev)))
    S = jnp.asarray(rng.normal(35.0, 0.3, size=(n_cells, nlev)))
    T_ref = T + 1.0
    S_ref = S - 0.25
    z = jnp.zeros_like(T)
    g1 = jnp.asarray(rng.uniform(0.0, 1e-5, size=(n_cells,)))
    g2 = jnp.asarray(rng.uniform(0.0, 1e-5, size=(n_cells, nlev)))
    dT1, _ = apply_sponge_tracer_relaxation(
        z, z, T, S, SpongeForcing(gamma=g1, T_ref=T_ref, S_ref=S_ref))
    np.testing.assert_array_equal(
        np.asarray(dT1), np.asarray(g1[:, None] * (T_ref - T)))
    dT2, _ = apply_sponge_tracer_relaxation(
        z, z, T, S, SpongeForcing(gamma=g2, T_ref=T_ref, S_ref=S_ref))
    np.testing.assert_array_equal(
        np.asarray(dT2), np.asarray(g2 * (T_ref - T)))


# ---------------------------------------------------------------------------
# Dispatch / guard mutations
# ---------------------------------------------------------------------------

def test_bad_gamma_rank_raises():
    T, S, T_ref, S_ref = _fields(9)
    z = jnp.zeros_like(T)
    g1 = jnp.full((_N_LAT,), 1e-5)   # rank T.ndim − 2: neither form
    with pytest.raises(ValueError, match="rank"):
        apply_sponge_tracer_relaxation(
            z, z, T, S, SpongeForcing(gamma=g1, T_ref=T_ref, S_ref=S_ref))


def test_gamma_3d_with_momentum_sponge_raises():
    """_bc_sponge_relaxation: a full-rank gamma + u_ref/v_ref is rejected at
    trace time (the momentum sponge needs the 2-D horizontal form)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_sponge_relaxation,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    T, S, T_ref, S_ref = _fields(10)
    g3 = jnp.full(T.shape, 1e-5)
    u = jnp.zeros((_N_LAT, _N_LON + 1, _NLEV))
    v = jnp.zeros((_N_LAT + 1, _N_LON, _NLEV))
    du = jnp.zeros_like(u); dv = jnp.zeros_like(v)
    z = jnp.zeros_like(T)
    sp = SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref, u_ref=u)
    with pytest.raises(ValueError, match="tracer relaxation only"):
        _bc_sponge_relaxation(du, dv, z, z, T, S, u, v, sp, grid)
    # 2-D gamma + u_ref is the supported momentum-sponge path — must not raise.
    sp2 = SpongeForcing(gamma=jnp.full((_N_LAT, _N_LON), 1e-5),
                        T_ref=T_ref, S_ref=S_ref, u_ref=u)
    _bc_sponge_relaxation(du, dv, z, z, T, S, u, v, sp2, grid)


def test_mpas_gamma_3d_with_momentum_sponge_raises():
    """MPAS path (mpas_ocean_baroclinic_tendencies): a full-rank (nCells, nlev)
    gamma combined with the edge-velocity sponge (u_ref) is rejected at trace
    time — the EXT-N1 guard mirroring the lat-lon path.  Without it, the
    momentum sponge averages gamma to edges and broadcasts over the vertical
    via ``gamma_edge[:, None]``, silently producing a ``(nEdges, nlev, nlev)``
    increment instead of ``(nEdges, nlev)``.  A 1-D (nCells,) gamma + u_ref is
    the supported momentum-sponge path and must NOT raise."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.dynamics.ocean_pe_mpas import (
        mpas_ocean_baroclinic_tendencies,
    )
    mesh = create_voronoi_mesh(subdivision_level=1)
    zc = create_ocean_z_star(n_levels=4, H_max=500.0, dz_surface=20.0,
                             dz_deep=200.0)
    st = rest_state_mpas_ocean(mesh, zc, T_water_init_C=15.0, T_deep=2.0,
                               S_uniform=35.0, H_max=500.0,
                               land_lat_threshold=85.0)
    cfg = MPASOceanConfig(A_h=1.0e3, K_h=1.0e2, n_barotropic_substeps=3)
    n_cells = st.T.data.shape[0]
    nlev = st.T.data.shape[1]
    n_edges = st.u.data.shape[0]
    u_ref = jnp.zeros((n_edges, nlev))
    g3 = jnp.full((n_cells, nlev), 1e-6)
    T_ref = st.T.data + 1.0
    S_ref = st.S.data - 0.25
    sp3 = SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref, u_ref=u_ref)
    with pytest.raises(ValueError, match="tracer relaxation only"):
        mpas_ocean_baroclinic_tendencies(st, mesh, zc, cfg, sponge=sp3)
    # 1-D gamma + u_ref: supported momentum-sponge path — must not raise.
    g1 = jnp.full((n_cells,), 1e-6)
    sp1 = SpongeForcing(gamma=g1, T_ref=T_ref, S_ref=S_ref, u_ref=u_ref)
    mpas_ocean_baroclinic_tendencies(st, mesh, zc, cfg, sponge=sp1)
    # 3-D gamma WITHOUT u_ref: tracer-only relaxation — must not raise.
    sp3t = SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref)
    mpas_ocean_baroclinic_tendencies(st, mesh, zc, cfg, sponge=sp3t)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

def test_grad_finite_through_3d_gamma():
    T, S, T_ref, S_ref = _fields(11)
    rng = np.random.default_rng(12)
    g3 = jnp.asarray(rng.uniform(0.0, 1e-5, size=T.shape))
    sp = SpongeForcing(gamma=g3, T_ref=T_ref, S_ref=S_ref)

    def loss(T_in):
        z = jnp.zeros_like(T_in)
        dT, dS = apply_sponge_tracer_relaxation(z, z, T_in, S, sp)
        return jnp.sum(dT ** 2) + jnp.sum(dS ** 2)

    g = jax.grad(loss)(T)
    assert bool(jnp.all(jnp.isfinite(g)))
    # Analytic: d/dT Σ(γ(T_ref−T))² = −2γ²(T_ref−T)
    np.testing.assert_allclose(
        np.asarray(g), np.asarray(-2.0 * g3 ** 2 * (T_ref - T)),
        rtol=1e-12, atol=0)
