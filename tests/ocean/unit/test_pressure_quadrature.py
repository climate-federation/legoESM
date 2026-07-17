"""p' vertical quadrature options in iterate_eos_and_pressure_anomaly:
legacy "cell_integral" vs NEMO dynhpg "nemo_trapezoid" (DINO L1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)

_G = constants.g
_RHO0 = 1026.0


def _run(rho_field, dz, quadrature, is_active=None):
    """Drive the helper with a fixed density field (identity EOS on T)."""
    T = jnp.asarray(rho_field)
    S = jnp.zeros_like(T)
    mask = jnp.ones(T.shape[:-1])
    _, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: f,
        lambda T_, S_, p_: T_,          # rho == T, no p dependence
        jnp.asarray(dz), _RHO0, _G,
        quadrature=quadrature, is_active_3d=is_active,
    )
    return np.asarray(rho_prime), np.asarray(p_prime)


def test_uniform_grid_rules_agree():
    """On a UNIFORM grid the two quadratures are algebraically equal."""
    rng = np.random.default_rng(0)
    rho = _RHO0 + rng.normal(size=(3, 4, 8))
    dz = np.full(8, 55.0)
    _, p_cell = _run(rho, dz, "cell_integral")
    _, p_trap = _run(rho, dz, "nemo_trapezoid")
    np.testing.assert_allclose(p_trap, p_cell, rtol=1e-13)


def test_stretched_grid_matches_f90_recurrence():
    """On a stretched grid the trapezoid must equal a literal
    transliteration of the dynhpg recurrence (and differ from legacy)."""
    rng = np.random.default_rng(1)
    nlev = 10
    dz = np.geomspace(10.0, 400.0, nlev)
    rho = _RHO0 + rng.normal(size=(2, 3, nlev))
    rho_prime, p_trap = _run(rho, dz, "nemo_trapezoid")

    # F90 transliteration: e3w(1)=dz(1); e3w(k)=(dz(k)+dz(k-1))/2;
    # P(1)=(g/2)*e3w(1)*rho'(1); P(k)=P(k-1)+(g/2)*e3w(k)*(rho'(k)+rho'(k-1))
    P = np.zeros_like(rho_prime)
    P[..., 0] = 0.5 * _G * dz[0] * rho_prime[..., 0]
    for k in range(1, nlev):
        e3w = 0.5 * (dz[k] + dz[k - 1])
        P[..., k] = (P[..., k - 1]
                     + 0.5 * _G * e3w
                     * (rho_prime[..., k] + rho_prime[..., k - 1]))
    np.testing.assert_allclose(p_trap, P, rtol=1e-13)

    _, p_cell = _run(rho, dz, "cell_integral")
    assert np.abs(p_trap - p_cell).max() > 0.0


def test_seafloor_mask_freezes_column():
    """With is_active_3d, a column's p' stays constant below its own
    bottom (NEMO's masked rhd)."""
    nlev = 8
    dz = np.geomspace(10.0, 200.0, nlev)
    rho = _RHO0 + np.linspace(1.0, 3.0, nlev)[None, None, :]
    rho = np.broadcast_to(rho, (2, 2, nlev)).copy()
    active = np.ones((2, 2, nlev), bool)
    active[0, 0, 5:] = False               # this column ends at k=4
    _, p = _run(rho, dz, "nemo_trapezoid", is_active=jnp.asarray(active))
    col = p[0, 0]
    # increments beyond the bottom pair-step are zero
    np.testing.assert_allclose(col[6:], col[6], rtol=0, atol=1e-9)


def test_uniform_anomaly_equals_g_rho_gdept():
    """Analytic hand-check (3 levels): a UNIFORM density anomaly ρ' with
    the NEMO e3w ladder gives p'(k) = g·ρ'·gdept(k) exactly.

    NEMO hpg_zco with e3w(1)=2·gdept(1), e3w(k)=gdept(k)−gdept(k−1) and a
    constant ρ' telescopes:
      p'(1) = (g/2)·2·gdept(1)·ρ' = g·ρ'·gdept(1)
      p'(k) = p'(k−1) + (g/2)·(gdept(k)−gdept(k−1))·(ρ'+ρ') = g·ρ'·gdept(k)
    """
    gdept = np.array([5.0, 15.0, 30.0])          # analytic (non-midpoint) T-depths
    dz = np.array([10.0, 10.0, 20.0])            # any consistent cell thicknesses
    c = 2.0
    rho = np.full((1, 1, 3), _RHO0 + c)

    T = jnp.asarray(rho)
    _, _, p = iterate_eos_and_pressure_anomaly(
        T, jnp.zeros_like(T), jnp.ones(T.shape[:-1]), lambda f: f,
        lambda T_, S_, p_: T_, jnp.asarray(dz), _RHO0, _G,
        quadrature="nemo_trapezoid", trapezoid_t_depth_1d=jnp.asarray(gdept))
    expected = _G * c * gdept
    np.testing.assert_allclose(np.asarray(p)[0, 0], expected, rtol=1e-13)


def test_z_star_t_depth_ref_plumbing():
    """create_z_star_from_thicknesses stores exact T-depths on
    ``t_depth_ref`` (None by default → midpoint z_full_ref unchanged)."""
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    dz = np.array([10.0, 10.0, 20.0, 40.0])
    zc_default = create_z_star_from_thicknesses(dz)
    assert zc_default.t_depth_ref is None            # default: no override

    gdept = np.array([4.9, 15.1, 30.2, 60.4])        # analytic, != midpoints
    zc = create_z_star_from_thicknesses(dz, t_depth_ref_m=gdept)
    # stored at the coordinate's policy-control dtype (like z_full_ref)
    np.testing.assert_allclose(np.asarray(zc.t_depth_ref), gdept, rtol=1e-6)
    # midpoint z_full_ref (and everything derived from it) is untouched
    np.testing.assert_array_equal(
        np.asarray(zc.z_full_ref), np.asarray(zc_default.z_full_ref))

    import pytest as _pytest
    with _pytest.raises(ValueError):
        create_z_star_from_thicknesses(dz, t_depth_ref_m=gdept[:3])   # wrong length


def test_unknown_quadrature_raises():
    rho = _RHO0 + np.zeros((2, 2, 4))
    with pytest.raises(ValueError, match="quadrature"):
        _run(rho, np.full(4, 50.0), "simpson")


def test_t_depth_ladder_form():
    """The t-depth-ladder e3w form equals the h-derived form for
    midpoint centres, and follows the ladder when centres are analytic
    (non-midpoint)."""
    rng = np.random.default_rng(2)
    nlev = 9
    dz = np.geomspace(12.0, 300.0, nlev)
    rho = _RHO0 + rng.normal(size=(2, 2, nlev))

    w = np.concatenate([[0.0], np.cumsum(dz)])
    t_mid = 0.5 * (w[:-1] + w[1:])

    T = jnp.asarray(rho)
    S = jnp.zeros_like(T)
    mask = jnp.ones(T.shape[:-1])

    def run(ladder):
        _, _, p = iterate_eos_and_pressure_anomaly(
            T, S, mask, lambda f: f, lambda T_, S_, p_: T_,
            jnp.asarray(dz), _RHO0, _G,
            quadrature="nemo_trapezoid",
            trapezoid_t_depth_1d=(None if ladder is None
                                  else jnp.asarray(ladder)))
        return np.asarray(p)

    np.testing.assert_allclose(run(t_mid), run(None), rtol=1e-12)

    # analytic-like centres (shifted off the midpoints): F90 recurrence
    # with e3w from the ladder
    t_ana = t_mid + np.linspace(0.5, 3.0, nlev)
    p_lad = run(t_ana)
    rho_prime = rho - _RHO0
    P = np.zeros_like(rho_prime)
    P[..., 0] = 0.5 * _G * (2.0 * t_ana[0]) * rho_prime[..., 0]
    for k in range(1, nlev):
        P[..., k] = (P[..., k - 1]
                     + 0.5 * _G * (t_ana[k] - t_ana[k - 1])
                     * (rho_prime[..., k] + rho_prime[..., k - 1]))
    np.testing.assert_allclose(p_lad, P, rtol=1e-12)


def test_pgf_caller_selects_t_depth_ref():
    """Exercise the production PGF caller's getattr-selection (the coverage gap):
    a z* coord WITH ``t_depth_ref`` uses NEMO's exact gdept ladder for the
    quadrature; WITHOUT it falls back to interface-midpoint ``|z_full_ref|``
    (byte-identical to pre-change). On a stretched grid the two must DIFFER —
    the ~0.5% depth-signed gap this change closes."""
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    rng = np.random.default_rng(7)
    nlev = 10
    dz = jnp.asarray(np.geomspace(10.0, 400.0, nlev))
    gdept = jnp.asarray(np.cumsum(np.asarray(dz)) - 0.5 * np.asarray(dz))
    rho = jnp.asarray(_RHO0 + rng.normal(size=(2, 3, nlev)))
    S = jnp.zeros_like(rho)
    mask = jnp.ones(rho.shape[:-1])

    zc_with = create_z_star_from_thicknesses(dz, t_depth_ref_m=gdept)
    zc_none = create_z_star_from_thicknesses(dz)
    assert zc_none.t_depth_ref is None                 # getattr -> fallback
    assert zc_with.t_depth_ref is not None

    def _pgf(zc):
        # the exact selection _bc_geometry_and_density performs:
        depth = (jnp.abs(zc.z_full_ref)
                 if getattr(zc, "t_depth_ref", None) is None
                 else jnp.asarray(zc.t_depth_ref))
        _, _, p = iterate_eos_and_pressure_anomaly(
            rho, S, mask, lambda f: f, lambda T_, S_, p_: T_,
            dz, _RHO0, _G, quadrature="nemo_trapezoid", trapezoid_t_depth_1d=depth)
        return np.asarray(p)

    p_with = _pgf(zc_with)
    p_none = _pgf(zc_none)
    # gdept ladder != midpoint ladder on a stretched grid -> pressures differ.
    assert not np.allclose(p_with, p_none, rtol=1e-6)
