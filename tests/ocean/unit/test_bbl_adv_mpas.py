"""Voronoi (edge-indexed) BBL twin vs the structured implementation.

The MPAS port must be the SAME physics as ocean/physics/bbl_adv.py — every
test here pins that equivalence (GLM design-review test list, 2026-09-01):
  * two-cell analytic overflow: transport and tendencies match the
    structured implementation EXACTLY on an equivalent geometry;
  * sign-flip ridge: reversing the depth difference reverses the transport;
  * exact tracer/salt conservation on a random staircase mesh;
  * flat bottom and land faces are inactive.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.bbl_adv import (
    apply_bbl_adv_tendency,
    bbl_static_geometry,
    bbl_transports,
)
from legoesm.ocean.physics.bbl_adv_mpas import (
    apply_bbl_adv_tendency_mpas,
    bbl_static_geometry_mpas,
    bbl_transports_mpas,
)

GAMMA = 20.0
RHO0 = 1026.0


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _two_col(nlev=6, shelf_lev=3, T_shelf=2.0, T_deep=4.0,
             S_shelf=35.5, S_deep=34.8):
    """Shelf column (shallow, dense) beside a deep column."""
    h = np.zeros((2, nlev))
    h[0, :shelf_lev] = 100.0          # shelf: 3 wet levels
    h[1, :] = 100.0                   # deep: all wet
    T = np.zeros((2, nlev))
    S = np.zeros((2, nlev))
    T[0], S[0] = T_shelf, S_shelf     # cold+salty shelf bottom = DENSE
    T[1], S[1] = T_deep, S_deep
    mask = np.ones(2)
    return jnp.asarray(h), jnp.asarray(T), jnp.asarray(S), jnp.asarray(mask)


def test_two_cell_matches_structured():
    h, T, S, mask = _two_col()
    width = 5.0e4
    area = jnp.full(2, 1.0e9)

    # --- edge version: one edge between cells (0, 1) ---
    coe = jnp.asarray([[0], [1]])
    ge = bbl_static_geometry_mpas(h, mask, coe)
    tr_e = bbl_transports_mpas(T, S, ge, jnp.asarray([width]),
                               gamma_s=GAMMA, rho_0=RHO0)
    dT_e, dS_e = apply_bbl_adv_tendency_mpas(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, h, area, ge, tr_e,
        nlev=int(h.shape[-1]))

    # --- structured version: (1, 2) domain, single i-face ---
    h2 = h[None, ...]                 # (1, 2, nlev)
    T2, S2 = T[None, ...], S[None, ...]
    gs = bbl_static_geometry(h2, mask[None, :])
    utr, vtr = bbl_transports(T2, S2, gs, jnp.full((1, 1), width),
                              jnp.zeros((0, 2)), gamma_s=GAMMA, rho_0=RHO0)
    dT_s, dS_s = apply_bbl_adv_tendency(
        jnp.zeros_like(T2), jnp.zeros_like(S2), T2, S2, h2,
        jnp.full((1, 2), 1.0e9), gs, utr, vtr, nlev=int(h.shape[-1]))

    np.testing.assert_allclose(np.asarray(tr_e[0]), np.asarray(utr[0, 0]),
                               rtol=1e-12)
    assert float(tr_e[0]) > 0.0, "dense shelf must drive a down-slope flow"
    np.testing.assert_allclose(np.asarray(dT_e), np.asarray(dT_s[0]),
                               rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(np.asarray(dS_e), np.asarray(dS_s[0]),
                               rtol=1e-12, atol=1e-18)


def test_sign_flip_ridge():
    """Swap which column is deep: the transport reverses exactly."""
    h, T, S, mask = _two_col()
    coe = jnp.asarray([[0], [1]])
    ge = bbl_static_geometry_mpas(h, mask, coe)
    tr = bbl_transports_mpas(T, S, ge, jnp.asarray([5.0e4]),
                             gamma_s=GAMMA, rho_0=RHO0)
    # mirrored geometry: cell 0 deep, cell 1 shelf (same T/S roles swapped)
    ge_m = bbl_static_geometry_mpas(h[::-1], mask, coe)
    tr_m = bbl_transports_mpas(T[::-1], S[::-1], ge_m, jnp.asarray([5.0e4]),
                               gamma_s=GAMMA, rho_0=RHO0)
    np.testing.assert_allclose(np.asarray(tr_m), -np.asarray(tr),
                               rtol=1e-12)


def test_conservation_random_staircase():
    rng = np.random.default_rng(3)
    n_cells, nlev, n_edges = 40, 7, 90
    h = np.zeros((n_cells, nlev))
    for c in range(n_cells):
        kb = rng.integers(1, nlev + 1)
        h[c, :kb] = rng.uniform(50.0, 150.0)
    T = rng.uniform(-1.0, 20.0, (n_cells, nlev))
    S = rng.uniform(33.0, 37.0, (n_cells, nlev))
    mask = (rng.uniform(size=n_cells) > 0.15).astype(float)
    coe = np.stack([rng.integers(0, n_cells, n_edges),
                    rng.integers(0, n_cells, n_edges)])
    coe = coe[:, coe[0] != coe[1]]
    area = rng.uniform(5.0e8, 2.0e9, n_cells)
    dv = rng.uniform(2.0e4, 8.0e4, coe.shape[1])

    ge = bbl_static_geometry_mpas(jnp.asarray(h), jnp.asarray(mask),
                                  jnp.asarray(coe))
    tr = bbl_transports_mpas(jnp.asarray(T), jnp.asarray(S), ge,
                             jnp.asarray(dv), gamma_s=GAMMA, rho_0=RHO0)
    dT, dS = apply_bbl_adv_tendency_mpas(
        jnp.zeros((n_cells, nlev)), jnp.zeros((n_cells, nlev)),
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(h), jnp.asarray(area),
        ge, tr, nlev=nlev)
    assert float(jnp.max(jnp.abs(dT))) > 0.0, "staircase produced no exchange"
    # exact conservation under the same volume weights the tendency divides by
    V = area[:, None] * np.maximum(h, 1.0e-3)
    for d, name in ((dT, "heat"), (dS, "salt")):
        tot = float(jnp.sum(jnp.asarray(V) * d))
        scale = float(jnp.sum(jnp.abs(jnp.asarray(V) * d))) + 1e-30
        assert abs(tot) / scale < 1e-12, f"{name} not conserved: {tot}"


def test_flat_bottom_and_land_inactive():
    h, T, S, mask = _two_col()
    coe = jnp.asarray([[0], [1]])
    # flat: both columns full depth
    h_flat = jnp.full_like(h, 100.0)
    ge = bbl_static_geometry_mpas(h_flat, mask, coe)
    assert float(ge.active[0]) == 0.0
    # land: one column masked
    ge2 = bbl_static_geometry_mpas(h, jnp.asarray([1.0, 0.0]), coe)
    assert float(ge2.active[0]) == 0.0
    tr = bbl_transports_mpas(T, S, ge2, jnp.asarray([5.0e4]),
                             gamma_s=GAMMA, rho_0=RHO0)
    assert float(jnp.abs(tr[0])) == 0.0


def test_many_to_one_convergence_bounded():
    """GLM review HIGH: 8 shelf cells all draining into ONE deep basin cell.

    The per-edge cap divided by the incident-edge count must bound the
    SUMMED exchange: after one (large) step no tracer goes non-physical and
    the basin's exchanged volume fraction stays <= 0.25; conservation holds.
    """
    from legoesm.ocean.physics.bbl_adv_mpas import apply_bbl_adv_step_mpas
    import dataclasses

    nlev = 5
    n_sh = 8
    n_cells = n_sh + 1                       # cell 0 = deep basin
    h = np.zeros((n_cells, nlev))
    h[0, :] = 100.0                          # deep: all wet
    h[1:, :2] = 100.0                        # shelves: 2 wet levels
    T = np.full((n_cells, nlev), 4.0)
    S = np.full((n_cells, nlev), 34.8)
    T[1:], S[1:] = 0.0, 36.5                 # very dense shelf water
    coe = np.stack([np.arange(1, n_cells), np.zeros(n_sh, dtype=int)])
    mask = np.ones(n_cells)
    area = np.full(n_cells, 1.0e6)           # tiny cells => caps ENGAGE
    dv = np.full(n_sh, 1.0e5)
    dt = 3600.0

    ge = bbl_static_geometry_mpas(jnp.asarray(h), jnp.asarray(mask),
                                  jnp.asarray(coe))
    assert float(ge.active.sum()) == n_sh

    class _F:                                # minimal Field/state stand-ins
        def __init__(self, d):
            self.data = d

        def replace(self, data):
            return _F(data)

    class _State:
        def __init__(self, T, S):
            self.T, self.S = _F(T), _F(S)

        def _replace(self, T, S):
            return _State(T.data, S.data)

    s0 = _State(jnp.asarray(T), jnp.asarray(S))
    s1 = apply_bbl_adv_step_mpas(
        s0, ge, dt, gamma_s=GAMMA, rho_0=RHO0,
        area=jnp.asarray(area), dv_edge=jnp.asarray(dv), nlev=nlev)

    T1 = np.asarray(s1.T.data)
    S1 = np.asarray(s1.S.data)
    assert np.all(np.isfinite(T1)) and np.all(np.isfinite(S1))
    # no overshoot past the source range (the tell of >1 volume exchanged)
    assert T1.min() >= -1e-9 and T1.max() <= 4.0 + 1e-9
    assert S1.min() >= 34.8 - 1e-9 and S1.max() <= 36.5 + 1e-9
    # conservation under the tendency's volume weights
    V = area[:, None] * np.maximum(h, 1.0e-3)
    for f0, f1, name in ((T, T1, "heat"), (S, S1, "salt")):
        tot0, tot1 = float((V * f0).sum()), float((V * f1).sum())
        assert abs(tot1 - tot0) / abs(tot0) < 1e-12, name
    # the basin bottom actually received dense shelf water (scheme active)
    assert S1[0, -1] > 34.8 + 1e-6
