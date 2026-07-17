"""Geometric-Boussinesq-depth EOS option (NEMO eos_insitu fidelity).

``iterate_eos_and_pressure_anomaly(eos_depth="geometric")`` and
``compute_ocean_rho(eos_depth="geometric")`` feed the EOS ``p = rho0*g*gdept``
so the Roquet EOS-80 ``zh`` term reconstructs the GEOMETRIC gdept exactly
(matches NEMO), instead of the ~0.5%-stretched in-situ hydrostatic integral.

Two things are pinned:
  1. geometric depth + matched rho0 == a direct ``nemo_roquet_eos(T,S,
     p=rho0*g*gdept, rho0)`` call to ~1e-12 on an analytic column;
  2. the DEFAULT ``insitu`` path is byte-identical to the pre-change code
     (an explicit inline transliteration of the old iteration).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.eos import make_eos_fn, nemo_roquet_eos, _ROQUET_EOS80
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)

_G = constants.g


def _analytic_column(nlev=20):
    """A wet analytic T/S column + NEMO-like stretched gdept ladder."""
    rng = np.random.default_rng(3)
    shape = (3, 4, nlev)
    T = 2.0 + 15.0 * rng.random(shape)          # 2..17 degC
    S = 34.0 + 2.0 * rng.random(shape)          # 34..36 PSU
    gdept = np.geomspace(5.0, 4500.0, nlev)     # positive-down, stretched
    dz = np.empty(nlev)
    dz[0] = 2.0 * gdept[0]
    dz[1:] = np.diff(gdept) * 2.0 - 0.0         # arbitrary; unused by geometric
    return jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept), jnp.asarray(dz)


def test_geometric_matches_direct_eos80_call():
    """geometric depth + matched rho0 == direct p=rho0*g*gdept EOS call."""
    T, S, gdept, dz = _analytic_column()
    rho0 = 1026.0
    mask = jnp.ones(T.shape[:-1])
    eos_fn = make_eos_fn("nemo_eos80", rho0=rho0)

    rho, _, _ = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: f, eos_fn, dz, rho0, _G,
        eos_depth="geometric", eos_geometric_depth_1d=gdept,
    )
    # Direct reference: exactly NEMO eos_insitu with geometric gdept.
    p_ref = rho0 * _G * gdept                    # broadcasts on trailing axis
    rho_ref = nemo_roquet_eos(T, S, p_ref, coeffs=_ROQUET_EOS80, rho0=rho0)
    np.testing.assert_allclose(np.asarray(rho), np.asarray(rho_ref), atol=1e-12)


def test_geometric_rho0_value_cancels():
    """Any consistent rho0 (make_eos_fn == p-construction) gives IDENTICAL
    density — the value cancels because zh recovers gdept exactly."""
    T, S, gdept, dz = _analytic_column()
    mask = jnp.ones(T.shape[:-1])

    def _geom(rho0):
        eos_fn = make_eos_fn("nemo_eos80", rho0=rho0)
        rho, _, _ = iterate_eos_and_pressure_anomaly(
            T, S, mask, lambda f: f, eos_fn, dz, rho0, _G,
            eos_depth="geometric", eos_geometric_depth_1d=gdept,
        )
        return np.asarray(rho)

    np.testing.assert_allclose(_geom(1026.0), _geom(1025.0), atol=1e-12)


def test_geometric_g_independent():
    """Regression: the geometric density must NOT depend on the passed config g.
    The EOS reconstructs depth as zh = p/(rho0*constants.g), so p_eos uses
    constants.g and g cancels -> zh = gdept exactly, independent of config g.
    A NEMO-fidelity config sets NEMO's gravity (!= constants.g); the density must
    still equal the exact-gdept reference. (Before the fix, using config.g for
    p_eos drifted ~5e-5 kg/m^3 here.)"""
    T, S, gdept, dz = _analytic_column()
    rho0 = 1026.0
    mask = jnp.ones(T.shape[:-1])
    eos_fn = make_eos_fn("nemo_eos80", rho0=rho0)
    g_nemo = 9.80665            # const-ok: NEMO grav, deliberately != constants.g
    assert abs(g_nemo - constants.g) > 1e-5
    rho, _, _ = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: f, eos_fn, dz, rho0, g_nemo,
        eos_depth="geometric", eos_geometric_depth_1d=gdept,
    )
    # exact-gdept reference uses constants.g (what the EOS uses internally)
    rho_ref = nemo_roquet_eos(
        T, S, rho0 * constants.g * gdept, coeffs=_ROQUET_EOS80, rho0=rho0)
    np.testing.assert_allclose(np.asarray(rho), np.asarray(rho_ref), atol=1e-12)


def test_compute_ocean_rho_geometric_matches_direct():
    """The PROBE path: compute_ocean_rho(eos_depth='geometric') (tendency_probe.py)
    feeds p = rho0*constants.g*gdept from z_coord.t_depth_ref and must equal the
    direct EOS call — same convention as the iterate path (both use constants.g)."""
    from types import SimpleNamespace

    from legoesm.ocean.eos import compute_ocean_rho
    from legoesm.ocean.vertical import create_z_star_from_thicknesses

    T, S, gdept, dz = _analytic_column()
    rho0 = 1026.0
    z_coord = create_z_star_from_thicknesses(dz, t_depth_ref_m=gdept)
    state = SimpleNamespace(T=SimpleNamespace(data=T), S=SimpleNamespace(data=S))
    eos_fn = make_eos_fn("nemo_eos80", rho0=rho0)
    # jacobian is unused by the geometric branch; pass a placeholder.
    rho = compute_ocean_rho(
        state, z_coord, jnp.ones(T.shape[:-1]), eos_fn=eos_fn,
        eos_depth="geometric", rho0=rho0)
    rho_ref = nemo_roquet_eos(
        T, S, rho0 * constants.g * gdept, coeffs=_ROQUET_EOS80, rho0=rho0)
    np.testing.assert_allclose(np.asarray(rho), np.asarray(rho_ref), atol=1e-12)


def test_insitu_default_byte_identical():
    """The default 'insitu' path reproduces the pre-change iteration exactly."""
    T, S, gdept, dz = _analytic_column()
    rho0 = 1025.0
    mask = jnp.ones(T.shape[:-1])
    eos_fn = make_eos_fn("nemo_eos80")       # default rho0 = module rho_0 (1025)

    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: f, eos_fn, dz, rho0, _G,   # eos_depth default
    )

    # Inline transliteration of the OLD iteration (2-pass in-situ hydrostatic
    # midpoint pressure, J=1, eta=0) — must match to the last bit.
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        compute_hydrostatic_pressure,
    )
    J_ref = jnp.ones(T.shape[:-1], dtype=T.dtype)
    eta_ref = jnp.zeros(T.shape[:-1], dtype=T.dtype)
    ref = eos_fn(T, S, jnp.zeros_like(T))
    for _ in range(2):
        p_h = compute_hydrostatic_pressure(ref, eta_ref, dz, J_ref, rho0, _G)
        ref = eos_fn(T, S, p_h)
    np.testing.assert_array_equal(np.asarray(rho), np.asarray(ref))

    # Geometric must DIFFER (the ~0.5% depth stretch is real), so the option
    # is non-vacuous.
    eos_fn_g = make_eos_fn("nemo_eos80", rho0=rho0)
    rho_g, _, _ = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: f, eos_fn_g, dz, rho0, _G,
        eos_depth="geometric", eos_geometric_depth_1d=gdept,
    )
    assert float(np.abs(np.asarray(rho) - np.asarray(rho_g)).max()) > 1e-4


def test_geometric_requires_depth_ladder():
    """geometric without the gdept ladder raises (dispatch hardening)."""
    T, S, gdept, dz = _analytic_column()
    mask = jnp.ones(T.shape[:-1])
    eos_fn = make_eos_fn("nemo_eos80", rho0=1026.0)
    try:
        iterate_eos_and_pressure_anomaly(
            T, S, mask, lambda f: f, eos_fn, dz, 1026.0, _G,
            eos_depth="geometric",   # no eos_geometric_depth_1d
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing gdept ladder")


def test_unknown_eos_depth_raises():
    T, S, gdept, dz = _analytic_column()
    mask = jnp.ones(T.shape[:-1])
    eos_fn = make_eos_fn("nemo_eos80")
    try:
        iterate_eos_and_pressure_anomaly(
            T, S, mask, lambda f: f, eos_fn, dz, 1025.0, _G,
            eos_depth="bogus",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown eos_depth")


def test_make_eos_fn_default_rho0_byte_identical():
    """make_eos_fn('nemo_eos80') default == explicit module-rho_0 build."""
    from legoesm.ocean.eos import rho_0 as _MOD_RHO0
    T, S, gdept, dz = _analytic_column()
    p = 1000.0 * jnp.ones_like(T)
    a = make_eos_fn("nemo_eos80")(T, S, p)
    b = make_eos_fn("nemo_eos80", rho0=_MOD_RHO0)(T, S, p)
    c = nemo_roquet_eos(T, S, p, coeffs=_ROQUET_EOS80)   # bare default
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(c))
