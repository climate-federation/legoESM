"""Advective bottom boundary layer (Campin-Goosse / NEMO trabbl nn_bbl_adv=2).

Idealized acceptance battery (the physics-contract test):
  * analytic 2-column dense-shelf overflow: transport magnitude matches the
    closed-form ``width*e3_bbl*g*gamma*max(0, drho/rho0)`` with the canonical
    EOS, sign is DOWN-slope, and a light shelf gives exactly zero;
  * exact tracer conservation: sum(area*h*d(pt)/dt) telescopes to 0 (fp tol);
  * tendency direction: the deep bottom cell moves TOWARD the dense shelf
    water (cools/salinifies for a cold-salty shelf);
  * flat bottom -> zero everywhere; land-adjacent faces inactive;
  * host wrapper integrates one Euler step and leaves land/dry cells alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    nemo_roquet_alpha_beta,
    wright_eos,
)
from legoesm.ocean.physics.bbl_adv import (
    apply_bbl_adv_tendency,
    bbl_static_geometry,
    bbl_transports,
)

RHO0 = 1025.0
GAMMA = 20.0


def _two_column_setup(nlev=6, dz=100.0, shelf_levels=2):
    """1x2 horizontal domain: column 0 = shelf (shelf_levels wet),
    column 1 = deep (all nlev wet).  Returns (h_ref, mask, T, S)."""
    h = np.zeros((1, 2, nlev))
    h[0, 0, :shelf_levels] = dz
    h[0, 1, :] = dz
    mask = np.ones((1, 2))
    T = np.full((1, 2, nlev), 10.0)
    S = np.full((1, 2, nlev), 35.0)
    return (jnp.asarray(h), jnp.asarray(mask),
            jnp.asarray(T), jnp.asarray(S))


def test_dense_shelf_transport_matches_closed_form():
    h, mask, T, S = _two_column_setup()
    # cold + salty shelf bottom -> denser than the deep neighbour
    T = T.at[0, 0, 1].set(4.0)
    S = S.at[0, 0, 1].set(36.0)
    geom = bbl_static_geometry(h, mask)
    dy_u = jnp.full((1, 1), 5.0e4)   # 50 km face
    dx_v = jnp.zeros((0, 2))
    utr, vtr = bbl_transports(T, S, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    # closed form: the EXACT NEMO eos_rab gating — alpha/beta per column at
    # ITS OWN bottom pressure, averaged across the face
    a_sh, b_sh = nemo_roquet_alpha_beta(
        jnp.asarray(4.0), jnp.asarray(36.0), jnp.asarray(150.0), rho0=RHO0)
    a_dp, b_dp = nemo_roquet_alpha_beta(
        jnp.asarray(10.0), jnp.asarray(35.0), jnp.asarray(550.0), rho0=RHO0)
    zgdrho = max(0.0, 0.5 * (a_sh + a_dp) * (10.0 - 4.0)
                 - 0.5 * (b_sh + b_dp) * (35.0 - 36.0))
    expect = 5.0e4 * 100.0 * constants.g * GAMMA * zgdrho * 1.0
    assert zgdrho > 0.0
    assert float(utr[0, 0]) == pytest.approx(expect, rel=1e-10)
    # sign: deeper column is i=1 -> mgrhu=+1 -> transport positive (down-slope)
    assert float(geom.mgrhu[0, 0]) == 1.0
    assert float(utr[0, 0]) > 0.0


def test_thermobaric_gating_uses_own_pressure_rab():
    """Steep-contrast regression (codex HIGH): the alpha/beta-at-own-pressure
    NEMO form differs measurably from a naive direct-density difference at
    the face-MEAN pressure — the implementation must follow the former."""
    nlev = 12
    h = np.zeros((1, 2, nlev))
    h[0, 0, :1] = 150.0                   # shelf: single 150 m cell
    h[0, 1, :] = 350.0                    # deep: 4200 m
    mask = jnp.ones((1, 2))
    T = jnp.asarray(np.full((1, 2, nlev), 2.0))
    S = jnp.asarray(np.full((1, 2, nlev), 34.7))
    T = T.at[0, 0, 0].set(1.0)            # slightly colder shelf
    S = S.at[0, 0, 0].set(34.75)          # slightly saltier shelf
    geom = bbl_static_geometry(jnp.asarray(h), mask)
    utr, _ = bbl_transports(T, S, geom, jnp.full((1, 1), 5.0e4),
                            jnp.zeros((0, 2)), gamma_s=GAMMA, rho_0=RHO0)
    # NEMO-form expectation
    dep_sh, dep_dp = 75.0, float(np.sum(h[0, 1]) - 0.5 * 350.0)
    a_sh, b_sh = nemo_roquet_alpha_beta(
        jnp.asarray(1.0), jnp.asarray(34.75), jnp.asarray(dep_sh), rho0=RHO0)
    a_dp, b_dp = nemo_roquet_alpha_beta(
        jnp.asarray(2.0), jnp.asarray(34.7), jnp.asarray(dep_dp), rho0=RHO0)
    a_bar = 0.5 * (float(a_sh) + float(a_dp))
    b_bar = 0.5 * (float(b_sh) + float(b_dp))
    zg_nemo = max(0.0, a_bar * (2.0 - 1.0) - b_bar * (34.7 - 34.75))
    expect = 5.0e4 * 150.0 * constants.g * GAMMA * zg_nemo
    assert float(utr[0, 0]) == pytest.approx(expect, rel=1e-10)
    # and the naive mean-pressure direct-density form is measurably DIFFERENT
    p_mean = RHO0 * constants.g * 0.5 * (dep_sh + dep_dp)
    drho_naive = float(wright_eos(jnp.asarray(1.0), jnp.asarray(34.75), jnp.asarray(p_mean))
                       - wright_eos(jnp.asarray(2.0), jnp.asarray(34.7), jnp.asarray(p_mean)))
    zg_naive = max(0.0, drho_naive / RHO0)
    assert abs(zg_naive - zg_nemo) > 1e-3 * max(zg_nemo, 1e-12)


def test_light_shelf_gives_zero_transport():
    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(20.0)                 # WARM shelf bottom -> lighter
    geom = bbl_static_geometry(h, mask)
    utr, _ = bbl_transports(T, S, geom, jnp.full((1, 1), 5.0e4),
                            jnp.zeros((0, 2)), gamma_s=GAMMA, rho_0=RHO0)
    assert float(utr[0, 0]) == 0.0


def test_flat_bottom_inactive():
    nlev = 5
    h = jnp.asarray(np.full((2, 3, nlev), 80.0))
    mask = jnp.ones((2, 3))
    geom = bbl_static_geometry(h, mask)
    assert float(jnp.sum(geom.u_active)) == 0.0
    assert float(jnp.sum(geom.v_active)) == 0.0


def test_land_adjacent_faces_inactive():
    h, mask, T, S = _two_column_setup()
    mask = mask.at[0, 0].set(0.0)
    geom = bbl_static_geometry(h, mask)
    assert float(jnp.sum(geom.u_active)) == 0.0


def test_exact_tracer_conservation_and_direction():
    """Random multi-column domain: the 3-leg cell conserves area*h-weighted
    tracer exactly; the deep bottom cell moves toward the shelf water."""
    rng = np.random.default_rng(11)
    ny, nx, nlev = 4, 6, 7
    # random staircase bathymetry, all wet
    nlev_col = rng.integers(2, nlev + 1, size=(ny, nx))
    h = np.zeros((ny, nx, nlev))
    for j in range(ny):
        for i in range(nx):
            h[j, i, :nlev_col[j, i]] = 90.0
    mask = np.ones((ny, nx))
    T = 8.0 + rng.standard_normal((ny, nx, nlev))
    S = 35.0 + 0.3 * rng.standard_normal((ny, nx, nlev))
    h_j, m_j = jnp.asarray(h), jnp.asarray(mask)
    T_j, S_j = jnp.asarray(T), jnp.asarray(S)
    geom = bbl_static_geometry(h_j, m_j)
    dy_u = jnp.full((ny, nx - 1), 4.0e4)
    dx_v = jnp.full((ny - 1, nx), 4.0e4)
    utr, vtr = bbl_transports(T_j, S_j, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    area = jnp.full((ny, nx), 1.6e9)
    dT, dS = apply_bbl_adv_tendency(
        jnp.zeros_like(T_j), jnp.zeros_like(S_j), T_j, S_j, h_j, area,
        geom, utr, vtr, nlev=nlev)
    w = np.asarray(area)[..., None] * np.maximum(h, 1.0e-3)
    # exact conservation (telescoping): compare against the LOCAL tendency
    # magnitude so the tolerance is scale-aware
    tot_T = float(np.sum(np.asarray(dT) * w))
    scale = float(np.sum(np.abs(np.asarray(dT)) * w)) + 1e-30
    assert abs(tot_T) < 1e-9 * scale
    tot_S = float(np.sum(np.asarray(dS) * w))
    scale_S = float(np.sum(np.abs(np.asarray(dS)) * w)) + 1e-30
    assert abs(tot_S) < 1e-9 * scale_S
    # at least one active face moved tracers
    assert scale > 0.0


def test_deep_bottom_moves_toward_shelf_water():
    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(2.0)                   # very cold dense shelf
    S = S.at[0, 0, 1].set(36.5)
    geom = bbl_static_geometry(h, mask)
    dy_u = jnp.full((1, 1), 5.0e4)
    dx_v = jnp.zeros((0, 2))
    utr, vtr = bbl_transports(T, S, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    area = jnp.full((1, 2), 1.0e9)
    dT, dS = apply_bbl_adv_tendency(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, h, area,
        geom, utr, vtr, nlev=6)
    # deep bottom (col 1, k=5) cools + salinifies toward the shelf water
    assert float(dT[0, 1, 5]) < 0.0
    assert float(dS[0, 1, 5]) > 0.0
    # shelf bottom (col 0, k=1) warms (receives deep water at shelf level)
    assert float(dT[0, 0, 1]) > 0.0


def test_host_wrapper_step_and_gating():
    from typing import NamedTuple

    from legoesm.ocean.physics.bbl_adv import apply_bbl_adv_step
    from legoesm.core.field import Field

    class _S(NamedTuple):  # minimal state stand-in (NamedTuple's own _replace)
        T: object
        S: object

    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(4.0)
    geom = bbl_static_geometry(h, mask)
    st = _S(T=Field(T, name="T"), S=Field(S, name="S"))
    out = apply_bbl_adv_step(
        st, geom, dt=600.0, gamma_s=GAMMA, rho_0=RHO0,
        area_2d=jnp.full((1, 2), 1.0e9),
        dy_u_faces=jnp.full((1, 1), 5.0e4),
        dx_v_faces=jnp.zeros((0, 2)), nlev=6)
    dT = np.asarray(out.T.data) - np.asarray(T)
    # exchange happened, bounded (|dT| << shelf-deep contrast), dry cells 0
    assert np.abs(dT).max() > 0.0
    assert np.abs(dT).max() < 6.0                  # << the 6 K contrast
    assert np.all(dT[0, 0, 2:] == 0.0)             # below shelf seafloor
