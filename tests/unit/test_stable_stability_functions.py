"""Selectable stable-regime Monin-Obukhov similarity functions (bulk_flux).

Covers the ``stability_scheme`` selector added to
:mod:`legoesm.core.bulk_flux` (``psi_m`` / ``psi_h`` / ``compute_most_fluxes``),
which chooses the STABLE-branch (zeta = z/L > 0) universal functions:

- ``"dyer1974"``              linear ``-5 zeta`` (default; must stay byte-identical)
- ``"beljaars_holtslag1991"`` Beljaars & Holtslag (1991)
- ``"grachev2007_sheba"``     Grachev et al. (2007) SHEBA
- ``"gryanik2020"``           Gryanik et al. (2020) modified SHEBA

The unstable branch (zeta < 0) is Businger-Dyer for every scheme.

Contracts exercised (see the FEATURE spec):
  * ``psi(zeta) -> 0`` as ``zeta -> 0`` for every scheme;
  * default ``"dyer1974"`` reproduces the pre-change ``-5 zeta`` EXACTLY (and the
    Businger-Dyer unstable branch), i.e. a byte-identical regression guard;
  * ``psi`` strictly decreasing (more negative) with increasing stable zeta;
  * an unknown scheme raises ``ValueError`` (dispatch hardening);
  * ``jax.grad`` of a flux w.r.t. an input state var is finite (no NaN) for each
    scheme in BOTH the stable and unstable regime;
  * at large stable zeta the non-linear schemes do NOT collapse the fluxes to
    zero the way linear ``-5 zeta`` does (they stay less negative than the
    unclipped Dyer value) yet retain suppression (more negative than a naive
    zeta-clipped linear value).

Run (worktree package edits are not the editable install):
  PYTHONPATH="$(pwd)/packages/core:$(pwd)/packages/coupler:$(pwd)/packages/land" \
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
    tests/unit/test_stable_stability_functions.py -x -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import (
    compute_most_fluxes,
    psi_h,
    psi_m,
    validate_stability_scheme,
)

# All selectable stable-regime schemes and the non-linear (non-default) subset.
SCHEMES = ("dyer1974", "beljaars_holtslag1991", "grachev2007_sheba", "gryanik2020")
NONLINEAR = tuple(s for s in SCHEMES if s != "dyer1974")


# ---------------------------------------------------------------------------
# Pre-change reference implementation (byte-identity regression guard).
# Verbatim copy of the ORIGINAL psi_m/psi_h stable=-5*zeta form so the default
# scheme is proven identical to today, independent of the new dispatch.
# ---------------------------------------------------------------------------
def _old_psi_m(zeta):
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)
    x = jnp.power(1.0 - 16.0 * zeta_neg, 0.25)
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x ** 2) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )
    stable = -5.0 * zeta_pos
    return jnp.where(zeta_c < 0.0, unstable, stable)


def _old_psi_h(zeta):
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)
    y = jnp.sqrt(1.0 - 16.0 * zeta_neg)
    unstable = 2.0 * jnp.log((1.0 + y) / 2.0)
    stable = -5.0 * zeta_pos
    return jnp.where(zeta_c < 0.0, unstable, stable)


# ---------------------------------------------------------------------------
# 1. zeta -> 0 limit: every scheme's psi_m, psi_h -> 0.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
def test_zeta_zero_limit(scheme):
    # Near-neutral stable zeta: psi must vanish (continuity with the unstable
    # branch, which also -> 0). Magnitude ~ slope * zeta ~ O(5e-8) at 1e-8.
    z8 = jnp.asarray(1e-8)
    assert abs(float(psi_m(z8, scheme))) < 1e-6
    assert abs(float(psi_h(z8, scheme))) < 1e-6
    # And it shrinks toward 0 as zeta shrinks (not merely small at one point).
    z4, z6 = jnp.asarray(1e-4), jnp.asarray(1e-6)
    for fn in (psi_m, psi_h):
        assert abs(float(fn(z6, scheme))) < abs(float(fn(z4, scheme)))
        assert abs(float(fn(z8, scheme))) < abs(float(fn(z6, scheme)))
    # Exactly zeta = 0 selects the stable branch (0 < 0 is False) -> ~0.
    assert abs(float(psi_m(jnp.asarray(0.0), scheme))) < 1e-6
    assert abs(float(psi_h(jnp.asarray(0.0), scheme))) < 1e-6


# ---------------------------------------------------------------------------
# 2. Default "dyer1974" reproduces the pre-change -5*zeta EXACTLY.
# ---------------------------------------------------------------------------
def test_dyer1974_is_byte_identical_default():
    zetas = jnp.asarray(
        [-8.0, -2.0, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 2.0, 5.0, 9.0]
    )
    # Default kwarg == explicit "dyer1974".
    np.testing.assert_array_equal(
        np.asarray(psi_m(zetas)), np.asarray(psi_m(zetas, "dyer1974"))
    )
    np.testing.assert_array_equal(
        np.asarray(psi_h(zetas)), np.asarray(psi_h(zetas, "dyer1974"))
    )
    # Default == the ORIGINAL implementation, bit-for-bit, on BOTH branches.
    np.testing.assert_array_equal(np.asarray(psi_m(zetas)), np.asarray(_old_psi_m(zetas)))
    np.testing.assert_array_equal(np.asarray(psi_h(zetas)), np.asarray(_old_psi_h(zetas)))


def test_dyer1974_stable_branch_is_minus_5_zeta():
    # Stable zeta in (0, 10): psi = -5*zeta exactly (no clip/floor active).
    zetas = jnp.asarray([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 9.0])
    expected = -5.0 * zetas
    np.testing.assert_array_equal(np.asarray(psi_m(zetas, "dyer1974")), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(psi_h(zetas, "dyer1974")), np.asarray(expected))


# ---------------------------------------------------------------------------
# 3. Monotonic decreasing (more negative) with increasing stable zeta.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
def test_monotonic_decreasing_stable(scheme):
    # Stay below the |zeta| <= 10 clip so the pure stable function is tested.
    zetas = jnp.linspace(0.01, 8.0, 80)
    pm = np.asarray(psi_m(zetas, scheme))
    ph = np.asarray(psi_h(zetas, scheme))
    assert np.all(np.diff(pm) < 0.0), f"{scheme} psi_m not strictly decreasing"
    assert np.all(np.diff(ph) < 0.0), f"{scheme} psi_h not strictly decreasing"


# ---------------------------------------------------------------------------
# 4. Unknown scheme -> ValueError (dispatch hardening).
# ---------------------------------------------------------------------------
def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        validate_stability_scheme("dyer1975")  # typo
    with pytest.raises(ValueError):
        psi_m(jnp.asarray(0.5), "nope")
    with pytest.raises(ValueError):
        psi_h(jnp.asarray(0.5), "nope")
    with pytest.raises(ValueError):
        compute_most_fluxes(
            jnp.asarray(5.0), jnp.asarray(0.0),
            jnp.asarray(290.0), jnp.asarray(0.008),
            jnp.asarray(285.0), jnp.asarray(0.010), jnp.asarray(1.2),
            scheme="most", n_iter=3, stability_scheme="bogus",
        )


def test_all_valid_schemes_accepted():
    for scheme in SCHEMES:
        validate_stability_scheme(scheme)  # must not raise


# ---------------------------------------------------------------------------
# 5. Differentiability: finite (no NaN) grads in BOTH regimes, every scheme.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", SCHEMES)
def test_psi_grads_finite_both_regimes(scheme):
    # Directly probe the AD-safe double-branch construction at unstable AND
    # stable zeta (including near-neutral, where the floors bite).
    for z0 in (-3.0, -0.5, -1e-3, 1e-3, 0.5, 3.0):
        gm = jax.grad(lambda z: psi_m(z, scheme))(jnp.asarray(z0))
        gh = jax.grad(lambda z: psi_h(z, scheme))(jnp.asarray(z0))
        assert jnp.isfinite(gm), f"{scheme}: dpsi_m/dzeta not finite at zeta={z0}"
        assert jnp.isfinite(gh), f"{scheme}: dpsi_h/dzeta not finite at zeta={z0}"


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("flux_scheme", ("most", "coare3"))
@pytest.mark.parametrize("T_sfc0", (284.0, 296.0))  # cold->stable, warm->unstable
def test_flux_grad_finite(scheme, flux_scheme, T_sfc0):
    q_atm = jnp.asarray(0.008)
    q_sfc = jnp.asarray(0.010)
    T_atm = jnp.asarray(290.0)
    rho = jnp.asarray(1.2)

    def loss(T_sfc):
        tau_x, _tau_y, shflx, lhflx, _ustar = compute_most_fluxes(
            jnp.asarray(6.0), jnp.asarray(1.0), T_atm, q_atm,
            T_sfc, q_sfc, rho,
            z_ref=10.0, z0_init=1e-4, scheme=flux_scheme, n_iter=6,
            stability_scheme=scheme,
        )
        return jnp.sum(shflx ** 2 + lhflx ** 2 + tau_x ** 2)

    g = jax.grad(loss)(jnp.asarray(T_sfc0))
    assert jnp.isfinite(g), (
        f"{scheme}/{flux_scheme}: d(flux)/dT_sfc not finite at T_sfc={T_sfc0}"
    )


# ---------------------------------------------------------------------------
# 6. Large stable zeta: non-linear schemes do not collapse fluxes to zero.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", NONLINEAR)
def test_large_stable_zeta_does_not_collapse(scheme):
    z = jnp.asarray(5.0)
    # Reference linear-Dyer values at the SAME zeta:
    dyer_unclipped = -5.0 * 5.0  # = -25 : over-suppresses -> flux -> 0
    dyer_clipped_at_1 = -5.0 * 1.0  # = -5 : a naive zeta-clipped linear value
    for fn in (psi_m, psi_h):
        val = float(fn(z, scheme))
        dyer = float(fn(z, "dyer1974"))  # = -25 at zeta=5
        # Less negative than the unclipped linear form -> smaller log-law
        # denominator -> larger transfer coefficient -> fluxes DON'T collapse.
        assert val > dyer, f"{scheme}: {fn.__name__}({float(z)})={val} !> Dyer {dyer}"
        assert dyer_unclipped < val < dyer_clipped_at_1, (
            f"{scheme}: {fn.__name__} = {val} outside expected ({dyer_unclipped}, "
            f"{dyer_clipped_at_1})"
        )


# ---------------------------------------------------------------------------
# Coupler parity fix: the ocean tile now accepts bulk_scheme="most" (previously
# it raised, unlike the sea-ice / land / lake tiles). Needs the land package on
# PYTHONPATH (coupler imports it); skip cleanly if unavailable.
# ---------------------------------------------------------------------------
def test_coupler_ocean_accepts_most():
    pytest.importorskip("legoesm.land.multilayer_land")
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.core.coupling_fields import AtmToSurface

    shape = (2, 3)

    def full(v):
        return jnp.full(shape, v)

    forcing = AtmToSurface(
        sw_down=full(200.0), lw_down=full(300.0),
        precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
        T_lowest=full(290.0), q_lowest=full(0.008),
        u_lowest=full(6.0), v_lowest=full(1.0),
        p_lowest=full(95000.0), p_surface=full(101325.0),
        rho_lowest=full(1.2), cos_zenith=full(0.5),
        co2_ppmv=jnp.asarray(400.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )
    # Cold SST -> stable stratification over the ocean (exercises the fixed-
    # roughness MOST stable branch).
    sst = full(286.0)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    config = CouplerConfig(bulk_scheme="most", z_ref=10.0, bulk_n_iter=4)
    resp = ocean_tile_response(forcing, sst, ocean_u, ocean_v, config)
    assert bool(jnp.all(jnp.isfinite(resp.shflx)))
    assert bool(jnp.all(jnp.isfinite(resp.lhflx)))

    # Differentiable through the coupler MOST "most" path.
    def coupler_loss(sst_in):
        r = ocean_tile_response(forcing, sst_in, ocean_u, ocean_v, config)
        return jnp.mean(r.shflx ** 2)

    grad = jax.grad(coupler_loss)(sst)
    assert bool(jnp.all(jnp.isfinite(grad)))
