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
    _DYER_STABLE_BETA,
    _DYER_UNSTABLE_GAMMA,
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


def test_coupler_stability_scheme_threads_to_fluxes():
    """CouplerConfig.stability_scheme reaches compute_most_fluxes: under a
    STABLE column (cold SST) the SHEBA stable functions give a different
    sensible-heat flux than the Dyer default, and an unknown name raises."""
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
        u_lowest=full(3.0), v_lowest=full(0.5),
        p_lowest=full(95000.0), p_surface=full(101325.0),
        rho_lowest=full(1.2), cos_zenith=full(0.5),
        co2_ppmv=jnp.asarray(400.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )
    sst = full(283.0)  # strongly stable: warm air over cold water
    zeros = jnp.zeros(shape)

    def shflx(stability):
        cfg = CouplerConfig(bulk_scheme="most", bulk_n_iter=6,
                            stability_scheme=stability)
        return np.asarray(
            ocean_tile_response(forcing, sst, zeros, zeros, cfg).shflx)

    dyer = shflx("dyer1974")
    sheba = shflx("grachev2007_sheba")
    assert np.all(np.isfinite(dyer)) and np.all(np.isfinite(sheba))
    assert not np.allclose(dyer, sheba), (
        "stability_scheme did not reach the coupler MOST fluxes")
    with pytest.raises(ValueError):
        shflx("dyer1975")


def test_sea_ice_stability_scheme_threads_to_fluxes():
    """SeaIceConfig.stability_scheme reaches the air-ice MOST fluxes (the
    Arctic/SHEBA use case): scheme choice changes the stable-column flux."""
    pytest.importorskip("legoesm.ice.sea_ice")
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import _bulk_flux_dispatch

    shape = (2, 3)

    def full(v):
        return jnp.full(shape, v)

    forcing = AtmToSurface(
        sw_down=full(50.0), lw_down=full(200.0),
        precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
        T_lowest=full(268.0), q_lowest=full(0.002),
        u_lowest=full(4.0), v_lowest=full(0.5),
        p_lowest=full(100000.0), p_surface=full(101325.0),
        rho_lowest=full(1.35), cos_zenith=full(0.2),
        co2_ppmv=jnp.asarray(400.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )
    T_ice = full(263.0)  # ice colder than air: stable surface layer

    def shflx(stability):
        cfg = SeaIceConfig(bulk_scheme="most", bulk_n_iter=6,
                           stability_scheme=stability)
        _, _, sh, _ = _bulk_flux_dispatch(T_ice, forcing, cfg, U_min=0.5)
        return np.asarray(sh)

    dyer = shflx("dyer1974")
    gryanik = shflx("gryanik2020")
    assert np.all(np.isfinite(dyer)) and np.all(np.isfinite(gryanik))
    assert not np.allclose(dyer, gryanik), (
        "stability_scheme did not reach the sea-ice MOST fluxes")


# ===========================================================================
# 7. Trainable Businger-Dyer coefficients (unstable_gamma / stable_beta):
#    neutral limit, threading direction, byte-compat, differentiability, and
#    the AIMIP param round-trip.  (feat/trainable-most-coeffs)
# ===========================================================================

# The module constants must stay at their historical values (the whole
# byte-compat contract rests on these defaults).
def test_module_coefficient_defaults_unchanged():
    assert _DYER_UNSTABLE_GAMMA == 16.0
    assert _DYER_STABLE_BETA == 5.0


# ---------------------------------------------------------------------------
# 7a. Neutral limit is scheme- AND coefficient-independent: psi(0)=0 for ANY
#     unstable_gamma / stable_beta.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gamma", (8.0, 16.0, 28.0))
@pytest.mark.parametrize("beta", (2.0, 5.0, 10.0))
def test_neutral_limit_coefficient_independent(gamma, beta):
    z0 = jnp.asarray(0.0)
    for fn in (psi_m, psi_h):
        assert abs(float(fn(z0, "dyer1974",
                            unstable_gamma=gamma, stable_beta=beta))) < 1e-6
    # And near-neutral on both sides shrinks toward 0 regardless of coeffs.
    for z in (jnp.asarray(-1e-8), jnp.asarray(1e-8)):
        for fn in (psi_m, psi_h):
            assert abs(float(fn(z, "dyer1974",
                                unstable_gamma=gamma, stable_beta=beta))) < 1e-5


# ---------------------------------------------------------------------------
# 7b. stable_beta scales the dyer1974 STABLE branch linearly: psi = -beta*zeta.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("beta", (2.0, 3.5, 5.0, 8.0, 10.0))
def test_stable_beta_scales_linearly(beta):
    zetas = jnp.asarray([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 9.0])
    expected = -beta * zetas
    for fn in (psi_m, psi_h):
        got = np.asarray(fn(zetas, "dyer1974", stable_beta=beta))
        np.testing.assert_allclose(got, np.asarray(expected), rtol=0, atol=0)


def test_stable_beta_larger_is_more_negative():
    # Sign/physics: a LARGER stable_beta => more negative psi (stronger stable
    # suppression => weaker fluxes).  Monotone in beta at fixed stable zeta.
    z = jnp.asarray(1.5)
    vals = [float(psi_m(z, "dyer1974", stable_beta=b)) for b in (2.0, 5.0, 10.0)]
    assert vals[0] > vals[1] > vals[2]  # increasing beta -> decreasing (more neg)


# ---------------------------------------------------------------------------
# 7c. unstable_gamma changes the UNSTABLE branch in the expected direction:
#     larger gamma => larger x/y => more POSITIVE psi (enhanced fluxes).
# ---------------------------------------------------------------------------
def test_unstable_gamma_larger_is_more_positive():
    z = jnp.asarray(-1.5)  # unstable
    pm = [float(psi_m(z, "dyer1974", unstable_gamma=g)) for g in (8.0, 16.0, 28.0)]
    ph = [float(psi_h(z, "dyer1974", unstable_gamma=g)) for g in (8.0, 16.0, 28.0)]
    assert pm[0] < pm[1] < pm[2], f"psi_m not increasing in gamma: {pm}"
    assert ph[0] < ph[1] < ph[2], f"psi_h not increasing in gamma: {ph}"


def test_unstable_gamma_only_touches_unstable_branch():
    # On the STABLE branch, unstable_gamma has NO effect (dyer1974 stable is
    # -beta*zeta, independent of gamma).
    z = jnp.asarray(1.2)
    a = float(psi_m(z, "dyer1974", unstable_gamma=8.0))
    b = float(psi_m(z, "dyer1974", unstable_gamma=28.0))
    assert a == b
    # And stable_beta has NO effect on the UNSTABLE branch.
    zn = jnp.asarray(-1.2)
    c = float(psi_h(zn, "dyer1974", stable_beta=2.0))
    d = float(psi_h(zn, "dyer1974", stable_beta=10.0))
    assert c == d


# ---------------------------------------------------------------------------
# 7d. Non-default coeffs do NOT leak into the non-linear stable schemes (they
#     carry their own published fits).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", NONLINEAR)
def test_stable_beta_ignored_by_nonlinear_schemes(scheme):
    z = jnp.asarray(2.0)
    base = float(psi_m(z, scheme))
    perturbed = float(psi_m(z, scheme, stable_beta=9.0))
    assert base == perturbed, f"{scheme} stable branch leaked stable_beta"


@pytest.mark.parametrize("scheme", NONLINEAR)
def test_unstable_gamma_ignored_by_nonlinear_schemes(scheme):
    # The non-linear stable schemes keep the HISTORICAL gamma=16 unstable branch:
    # a tuned unstable_gamma must NOT perturb their zeta<0 side (codex R1).
    zn = jnp.asarray(-2.0)
    for fn in (psi_m, psi_h):
        base = float(fn(zn, scheme))
        perturbed = float(fn(zn, scheme, unstable_gamma=26.0))
        assert base == perturbed, (
            f"{scheme} unstable branch leaked unstable_gamma ({fn.__name__})")
        # ...and it still equals the byte-identical historical (gamma=16) value.
        hist = float(fn(zn, scheme, unstable_gamma=16.0))
        assert base == hist


def test_coare_ignores_trainable_coefficients():
    # COARE uses psi_m_coare / psi_h_coare (its own Fairall coefficients); the
    # MOST-solver coare3 path must be byte-identical regardless of the Dyer
    # coeffs (codex R1 / coverage).
    args = (
        jnp.asarray(6.0), jnp.asarray(1.0),
        jnp.asarray(290.0), jnp.asarray(0.008),
        jnp.asarray(284.0), jnp.asarray(0.010), jnp.asarray(1.2),
    )
    kw = dict(z_ref=10.0, z0_init=1e-4, scheme="coare3", n_iter=6)
    base = compute_most_fluxes(*args, **kw)
    perturbed = compute_most_fluxes(
        *args, **kw, unstable_gamma=26.0, stable_beta=9.0)
    for b, p in zip(base, perturbed):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(p))


# ---------------------------------------------------------------------------
# 7e. Backward compatibility: omitting the new kwargs is byte-identical to the
#     literal 16 / 5 forms, for psi_m/psi_h AND compute_most_fluxes.
# ---------------------------------------------------------------------------
def test_psi_default_kwargs_byte_identical():
    zetas = jnp.asarray([-8.0, -2.0, -0.1, 0.0, 0.1, 2.0, 9.0])
    for fn in (psi_m, psi_h):
        default = np.asarray(fn(zetas, "dyer1974"))
        explicit = np.asarray(fn(zetas, "dyer1974",
                                 unstable_gamma=16.0, stable_beta=5.0))
        np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("flux_scheme", ("most", "large_yeager", "coare3"))
@pytest.mark.parametrize("T_sfc0", (283.0, 296.0))
def test_compute_most_fluxes_default_coeffs_byte_identical(flux_scheme, T_sfc0):
    args = (
        jnp.asarray(6.0), jnp.asarray(1.0),
        jnp.asarray(290.0), jnp.asarray(0.008),
        jnp.asarray(T_sfc0), jnp.asarray(0.010), jnp.asarray(1.2),
    )
    kw = dict(z_ref=10.0, z0_init=1e-4, scheme=flux_scheme, n_iter=6)
    base = compute_most_fluxes(*args, **kw)
    explicit = compute_most_fluxes(
        *args, **kw, unstable_gamma=16.0, stable_beta=5.0)
    for b, e in zip(base, explicit):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(e))


def test_compute_most_fluxes_coeffs_change_fluxes():
    # A non-default coeff must actually move the MOST fluxes on the Businger-Dyer
    # (non-COARE) path.  Stable column (cold SST) -> stable_beta matters.
    args_stable = (
        jnp.asarray(4.0), jnp.asarray(0.5),
        jnp.asarray(290.0), jnp.asarray(0.008),
        jnp.asarray(283.0), jnp.asarray(0.010), jnp.asarray(1.2),
    )
    kw = dict(z_ref=10.0, z0_init=1e-4, scheme="large_yeager", n_iter=8)
    base = compute_most_fluxes(*args_stable, **kw)
    strong = compute_most_fluxes(*args_stable, **kw, stable_beta=9.0)
    # Larger beta => stronger stable suppression => |shflx| smaller (weaker flux).
    assert not np.allclose(np.asarray(base[2]), np.asarray(strong[2]))
    assert abs(float(strong[2])) < abs(float(base[2])), (
        "larger stable_beta should weaken the stable sensible-heat flux")
    # Unstable column (warm SST) -> unstable_gamma matters.
    args_unstable = (
        jnp.asarray(4.0), jnp.asarray(0.5),
        jnp.asarray(290.0), jnp.asarray(0.008),
        jnp.asarray(298.0), jnp.asarray(0.012), jnp.asarray(1.2),
    )
    base_u = compute_most_fluxes(*args_unstable, **kw)
    strong_u = compute_most_fluxes(*args_unstable, **kw, unstable_gamma=26.0)
    assert not np.allclose(np.asarray(base_u[2]), np.asarray(strong_u[2]))


def test_compute_most_fluxes_positional_return_convergence_still_works():
    # ``return_convergence`` kept its historical POSITIONAL slot (the new coeffs
    # are keyword-only AFTER it), so a full-positional caller passing
    # return_convergence positionally is unbroken (codex R4).
    args = (
        jnp.asarray(6.0), jnp.asarray(1.0),
        jnp.asarray(290.0), jnp.asarray(0.008),
        jnp.asarray(285.0), jnp.asarray(0.010), jnp.asarray(1.2),
    )
    # positional order after (u,v,T,q,T_sfc,q_sfc,rho): z_ref, z_t, z_q, z0_init,
    # scheme, n_iter, charnock, L_latent, thermo_convention, gustiness_w_zi,
    # gustiness_beta, return_2m, z_diag, max_exchange_coeff, stability_scheme,
    # return_convergence
    out = compute_most_fluxes(
        *args,
        10.0, None, None, 1e-4, "large_yeager", 6, 0.011, None, "legoesm",
        None, 1.25, False, 2.0, None, "dyer1974", True,  # return_convergence=True
    )
    assert len(out) == 6  # tau_x, tau_y, sh, lh, ustar, most_residual
    assert np.all(np.isfinite(np.asarray(out[-1])))


def test_return_2m_uses_trained_coefficients():
    # return_2m recomputes psi_h for the 2 m diagnostic; it must use the SAME
    # (possibly non-default) coefficients as the converged profile (codex R2).
    #
    # This is an END-TO-END check: it confirms the returned T_2m responds to
    # stable_beta at all (a diagnostic hard-wired to default coeffs would still
    # move via the converged scales, so this alone does not ISOLATE the
    # diagnostic psi_h — see test_return_2m_diagnostic_psih_is_beta_sensitive
    # below for the isolated proof of the R2 fix).
    args = (
        jnp.asarray(6.0), jnp.asarray(1.0),
        jnp.asarray(290.0), jnp.asarray(0.008),
        jnp.asarray(283.0), jnp.asarray(0.010), jnp.asarray(1.2),  # stable
    )
    kw = dict(z_ref=10.0, z_t=2.0, z_q=2.0, z0_init=1e-4,
              scheme="large_yeager", n_iter=8, return_2m=True)
    base = compute_most_fluxes(*args, **kw)
    strong = compute_most_fluxes(*args, **kw, stable_beta=9.0)
    T2m_base = float(base[5])
    T2m_strong = float(strong[5])
    assert np.isfinite(T2m_base) and np.isfinite(T2m_strong)
    assert T2m_base != T2m_strong, (
        "return_2m diagnostic ignored the trained stable_beta")


def test_return_2m_diagnostic_psih_is_beta_sensitive():
    """ISOLATED R2 proof: the 2 m diagnostic evaluates psi_h at the diagnostic
    height with the SAME stable_beta as the solve.

    The diagnostic is ``T_2m = T_sfc - (theta*/kappa)*(ln(z_diag/z0_t) -
    psi_h(z_diag/L))`` (bulk_flux.py return_2m block).  We isolate the psi_h
    dependence from the converged scales by driving the converged solve to be
    (near-)beta-INDEPENDENT — a near-neutral column with n_iter high — so the
    total T_2m response is dominated by the diagnostic psi_h term.  Here the
    surface (289 K) is COLDER than the air (290 K) [stable], so theta* < 0 and
    ``T_2m = T_sfc - (theta*/kappa)*(ln(z_diag/z0_t) - psi_h(z_diag/L))`` with
    ``psi_h(zeta, dyer1974, beta) = -beta*zeta`` (beta>0, zeta>0).  A LARGER beta
    => more negative psi_h => larger ``(ln - psi_h)`` => (theta*<0) T_2m is
    pulled UP toward the (warmer) air, i.e. T_2m strictly INCREASES with beta.
    A diagnostic hard-wired to default beta would NOT show this monotone-in-beta
    ordering once the converged scales are held ~fixed.  (Direction verified
    numerically: T_2m(2,5,9) ~= 289.5923, 289.5926, 289.5930.)
    """
    # Weakly stable column (small T_sfc-T_atm gap) so the converged u*/theta*
    # barely move with beta, but zeta>0 so the diagnostic psi_h is active.
    args = (
        jnp.asarray(8.0), jnp.asarray(0.0),   # strong wind -> well-converged u*
        jnp.asarray(290.0), jnp.asarray(0.009),
        jnp.asarray(289.0), jnp.asarray(0.010), jnp.asarray(1.2),  # gap = 1 K
    )
    kw = dict(z_ref=10.0, z_t=2.0, z_q=2.0, z0_init=1e-4,
              scheme="large_yeager", n_iter=20, return_2m=True)
    T2m = [float(compute_most_fluxes(*args, **kw, stable_beta=b)[5])
           for b in (2.0, 5.0, 9.0)]
    assert all(np.isfinite(T2m))
    # Strictly increasing in beta (see docstring derivation).
    assert T2m[0] < T2m[1] < T2m[2], (
        f"2 m diagnostic psi_h not monotone in stable_beta: {T2m}")


# ---------------------------------------------------------------------------
# 7f. Differentiability: jax.grad of a MOST flux w.r.t. unstable_gamma AND
#     stable_beta is finite and NON-ZERO under a stability-dependent scheme.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flux_scheme", ("most", "large_yeager"))
def test_grad_wrt_coefficients_finite_nonzero(flux_scheme):
    # Use a column with BOTH stable and unstable content by taking two columns.
    u = jnp.asarray([5.0, 5.0])
    v = jnp.asarray([1.0, 1.0])
    T_atm = jnp.asarray([290.0, 290.0])
    q_atm = jnp.asarray([0.008, 0.008])
    T_sfc = jnp.asarray([283.0, 298.0])  # col0 stable, col1 unstable
    q_sfc = jnp.asarray([0.010, 0.013])
    rho = jnp.asarray([1.2, 1.2])

    def loss(gamma, beta):
        out = compute_most_fluxes(
            u, v, T_atm, q_atm, T_sfc, q_sfc, rho,
            z_ref=10.0, z0_init=1e-4, scheme=flux_scheme, n_iter=8,
            unstable_gamma=gamma, stable_beta=beta,
        )
        # sum over stress + sensible + latent
        return jnp.sum(out[0] ** 2 + out[2] ** 2 + out[3] ** 2)

    g_gamma = jax.grad(loss, argnums=0)(jnp.asarray(16.0), jnp.asarray(5.0))
    g_beta = jax.grad(loss, argnums=1)(jnp.asarray(16.0), jnp.asarray(5.0))
    assert jnp.isfinite(g_gamma) and jnp.isfinite(g_beta)
    assert abs(float(g_gamma)) > 0.0, "d(flux)/d(unstable_gamma) is exactly zero"
    assert abs(float(g_beta)) > 0.0, "d(flux)/d(stable_beta) is exactly zero"


# ---------------------------------------------------------------------------
# 7g. Config threading: SurfaceLayerConfig.most_* reach compute_surface_fluxes.
# ---------------------------------------------------------------------------
def test_surface_layer_config_threads_coefficients():
    pytest.importorskip("legoesm.atmosphere.physics.turbulence.surface_layer")
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )

    n = 4
    u = jnp.full((n,), 4.0)
    v = jnp.full((n,), 0.5)
    T = jnp.full((n,), 290.0)
    q_v = jnp.full((n,), 0.008)
    T_sfc = jnp.full((n,), 283.0)  # stable
    q_sfc = jnp.full((n,), 0.010)
    rho = jnp.full((n,), 1.2)

    def sh(beta):
        cfg = SurfaceLayerConfig(
            bulk_scheme="large_yeager", bulk_n_iter=8, most_stable_beta=beta,
        )
        return np.asarray(
            compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg)[2])

    base = sh(5.0)
    strong = sh(9.0)
    assert np.all(np.isfinite(base)) and np.all(np.isfinite(strong))
    assert not np.allclose(base, strong), (
        "SurfaceLayerConfig.most_stable_beta did not reach compute_most_fluxes")

    # Default config (constant scheme) is UNAFFECTED by the coefficients (the
    # constant path never touches MOST) -> confirm no accidental coupling.
    def sh_constant(beta):
        cfg = SurfaceLayerConfig(bulk_scheme="constant", most_stable_beta=beta)
        return np.asarray(
            compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg)[2])

    np.testing.assert_array_equal(sh_constant(2.0), sh_constant(10.0))


# ---------------------------------------------------------------------------
# 7h. AIMIP param round-trip: surface_most_* flow through AIMIPClassicalParams
#     -> to_surface_config -> SurfaceLayerConfig, within their bounds.
# ---------------------------------------------------------------------------
def test_aimip_param_roundtrip_most_coeffs():
    pytest.importorskip("legoesm.training.aimip_params")
    from legoesm.training.aimip_params import AIMIPClassicalParams

    params = AIMIPClassicalParams.from_defaults()
    d = params.as_dict()
    assert "surface_most_unstable_gamma" in d
    assert "surface_most_stable_beta" in d

    cfg = params.to_surface_config()
    # The trained (constrained) values land on the config fields.
    assert float(cfg.most_unstable_gamma) == pytest.approx(
        float(d["surface_most_unstable_gamma"]), rel=1e-6)
    assert float(cfg.most_stable_beta) == pytest.approx(
        float(d["surface_most_stable_beta"]), rel=1e-6)
    # And they sit inside the declared bounds.
    assert 8.0 <= float(cfg.most_unstable_gamma) <= 28.0
    assert 2.0 <= float(cfg.most_stable_beta) <= 10.0

    # A perturbed raw leaf moves the config value (genuinely trainable, not a
    # frozen default).
    import jax.numpy as _jnp
    raw2 = dict(params.raw_values)
    raw2["surface_most_stable_beta"] = raw2["surface_most_stable_beta"] + 1.0
    params2 = AIMIPClassicalParams(
        raw_values=raw2, constraints=params.constraints,
        spatial_surface=params.spatial_surface,
    )
    assert float(params2.to_surface_config().most_stable_beta) != float(
        cfg.most_stable_beta)
