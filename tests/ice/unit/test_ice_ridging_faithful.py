"""Mechanical ridging ORACLE-FAITHFULNESS + conservation tests.

``ice/ridging.py`` implements the Lipscomb (2007) ridging closure and had NO
direct test.  These pin:

  * the participation function ``participation_weights`` (Lipscomb 2007 eq. 22,
    ``b_k = a_k exp(-h_k/e*) / sum``) to round-off (rel 1e-9) against an
    independent scalar reimplementation;
  * the Hibler (1980) / Lipscomb (2007) eq. 26 ridge-thickness range
    ``H_min = 2 h_part``, ``H_max = min(mu_rdg sqrt(h_part), H_star)`` — via a
    single-donor column with analytic ``h_part``, checking the PER-RECEIVER-BIN
    area and volume against the independent uniform-``g`` overlap integral (pins
    the distribution, not just its support and mean), the ``H_star`` cap, and the
    two numerical regularizations (``H_max`` floor and over-thick collapse);
  * the DEFINING conservation invariants of ``apply_ridging`` (the truth tier
    that outranks oracle-matching): ice volume and bulk salt mass are conserved,
    total area is reduced by exactly ``closing_rate*dt`` (the CICE aksum
    normalisation) until area-limited, the donor snow is split by
    ``snow_fraction_retained`` with the deficit reported to the ocean, ridging
    pond water fully drains, and a divergent column is a no-op.

Independence: e_star / mu_rdg / H_star / snow_fraction_retained are local ``_O_*``
literals canaried against ``RidgingConfig`` (config == ``_O_*`` == value), and a
default-vs-explicit test proves the production defaults take those values; salt
mass uses ``constants.rho_ice`` and the ``_PSU_TO_FRACTION`` unit factor canaried
against the module.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the rel-1e-9 pins; restore the process-entry state in
    finally so selecting a single test never leaks x64 into another module."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


from legoesm import constants                                          # noqa: E402
from legoesm.ice.ridging import (                                      # noqa: E402
    participation_weights, apply_ridging, _PSU_TO_FRACTION,
)
from legoesm.ice.config import RidgingConfig                           # noqa: E402
from legoesm.ice.itd import category_bounds, upper_bounds              # noqa: E402

# Independent oracle literals (canaried in test_ridging_constants_match_config).
_O_E_STAR = 0.36          # participation e-folding thickness [m]
_O_MU_RDG = 4.0           # ridge-thickness multiplier
_O_H_STAR = 100.0         # maximum ridge thickness scale [m]
_O_SNOW_RETAINED = 0.5    # fraction of donor snow retained in ridge
_O_PSU_TO_FRACTION = 1.0e-3   # PSU (g/kg) -> mass fraction (kg/kg)


def _participation_oracle(a, h, e_star):
    e = max(e_star, 1.0e-3)
    raw = [float(a[k]) * math.exp(-float(h[k]) / e) for k in range(len(a))]
    total = sum(raw)
    if total <= 1.0e-30:
        return [0.0] * len(a)
    return [r / total for r in raw]


def _f(x):
    return jnp.array(x, dtype=jnp.float64)


# --- participation function (Lipscomb 2007 eq. 22) -----------------------------

@pytest.mark.parametrize("a,h", [
    ([0.3, 0.25, 0.2, 0.1, 0.05], [0.2, 0.6, 1.5, 3.0, 6.0]),
    ([0.5, 0.3, 0.15, 0.05, 0.0], [0.1, 0.4, 1.0, 2.5, 5.0]),
    ([0.1, 0.1, 0.1, 0.1, 0.1],   [0.3, 0.9, 1.8, 3.5, 7.0]),
])
def test_participation_matches_lipscomb_eq22(a, h):
    """participation_weights == a_k exp(-h_k/e*) / sum, to round-off."""
    got = participation_weights(_f(a), _f(h), _O_E_STAR)
    exp = _participation_oracle(a, h, _O_E_STAR)
    for k in range(len(a)):
        assert float(got[k]) == pytest.approx(exp[k], rel=1e-9, abs=1e-300)


def test_participation_normalised_thin_preferential():
    """Weights sum to 1 with ice and ridge thin ice preferentially (w/a strictly
    decreasing in h)."""
    a = _f([0.3, 0.25, 0.2, 0.1, 0.05])
    h = _f([0.2, 0.6, 1.5, 3.0, 6.0])
    w = participation_weights(a, h, _O_E_STAR)
    assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-12)
    wa = [float(w[k]) / float(a[k]) for k in range(5)]
    assert all(wa[k] > wa[k + 1] for k in range(4))           # thinner ridges more


def test_participation_zero_guard_ice_free_and_subthreshold():
    """The total>1e-30 guard returns exactly zero weights for BOTH an ice-free
    column (total == 0) and a nonzero-but-sub-threshold column (0 < total <=
    1e-30, here vanishing areas)."""
    h = _f([0.2, 0.6, 1.5, 3.0, 6.0])
    assert float(jnp.sum(participation_weights(jnp.zeros(5), h, _O_E_STAR))) == 0.0
    tiny = _f([1e-32, 1e-32, 1e-32, 1e-32, 1e-32])            # sum(a*exp) < 1e-30
    w_tiny = participation_weights(tiny, h, _O_E_STAR)
    assert float(jnp.sum(jnp.abs(w_tiny))) == 0.0             # guard fires, not raw/total


def test_participation_e_star_divide_floor():
    """e_star is floored at 1e-3 m before the divide (a smaller value would give
    different weights without the floor)."""
    a = _f([0.4, 0.3, 0.2, 0.1, 0.0])
    h = _f([0.1, 0.4, 1.0, 2.5, 5.0])
    w_tiny = participation_weights(a, h, 1e-4)                 # below the 1e-3 floor
    w_floor = participation_weights(a, h, 1e-3)
    for k in range(5):
        assert float(w_tiny[k]) == pytest.approx(float(w_floor[k]), rel=1e-12, abs=1e-300)


# --- conservation invariants (truth tier) --------------------------------------

def _column():
    n_cat = 5
    a_cat = _f([[0.3, 0.25, 0.2, 0.1, 0.05]])
    h_cat = _f([[0.2, 0.6, 1.5, 3.0, 6.0]])
    V_snow = _f([[0.03, 0.02, 0.01, 0.005, 0.002]])
    S_ice = _f([[5.0, 5.0, 4.0, 3.0, 2.0]])
    V_pond = _f([[0.01, 0.008, 0.0, 0.0, 0.0]])
    return n_cat, a_cat, h_cat, V_snow, S_ice, V_pond


def _apply(closing, dt=3600.0):
    n_cat, a_cat, h_cat, V_snow, S_ice, V_pond = _column()
    res = apply_ridging(a_cat, h_cat, V_snow, S_ice, _f([closing]), n_cat, dt,
                        V_pond_cat=V_pond, e_star=_O_E_STAR, mu_rdg=_O_MU_RDG,
                        H_star=_O_H_STAR, snow_fraction_retained=_O_SNOW_RETAINED)
    return (n_cat, a_cat, h_cat, V_snow, S_ice, V_pond), res


def test_ridging_conserves_ice_volume():
    (_, a_cat, h_cat, *_), res = _apply(0.5e-6)
    V_old = float(jnp.sum(h_cat * a_cat))
    V_new = float(jnp.sum(res["h"] * res["a"]))
    assert V_new == pytest.approx(V_old, rel=1e-12, abs=0.0)


def test_ridging_area_closing_matches_requested():
    """Total area is reduced by exactly closing_rate*dt (CICE aksum), when the
    column is not area-limited."""
    closing, dt = 0.5e-6, 3600.0
    (_, a_cat, *_), res = _apply(closing, dt)
    dA = float(jnp.sum(res["a"]) - jnp.sum(a_cat))
    assert dA == pytest.approx(-closing * dt, rel=1e-9, abs=0.0)


def test_ridging_area_saturates_when_all_ice_ridged():
    """A huge closing_rate cannot remove more than the available area: the net
    closing SATURATES (|dA| < requested) and the area stays finite and > 0."""
    closing, dt = 1.0e-2, 3600.0                              # enormous request
    (_, a_cat, *_), res = _apply(closing, dt)
    A_old = float(jnp.sum(a_cat))
    A_new = float(jnp.sum(res["a"]))
    assert 0.0 < A_new < A_old                                # ice compressed, not gone
    assert abs(A_new - A_old) < closing * dt                  # saturated below the request
    assert bool(jnp.all(jnp.isfinite(res["a"])))


def test_ridging_partial_cap_underdelivers_but_conserves_volume():
    """When a thin, tiny-area category DOMINATES the participation weight, its
    per-cat draw is area-capped (the compression is estimated before clipping).
    Here cat0 (weight ~0.81, area 0.001) is fully consumed — proof its requested
    draw exceeded its area — so the net closing UNDER-DELIVERS (|dA| well below
    the requested closing_rate*dt), substantiating the aksum per-cat caveat; the
    redistribution stays volume-conserving and finite."""
    n_cat = 5
    a_cat = _f([[0.001, 0.05, 0.05, 0.0, 0.0]])               # cat0 thin -> dominant weight
    h_cat = _f([[0.05, 2.0, 3.0, 0.0, 0.0]])                  # others thick -> tiny weight
    Z = jnp.zeros((1, n_cat))
    closing, dt = 2.0e-6, 3600.0
    # cat0 carries the majority of the participation weight
    w = participation_weights(a_cat[0], h_cat[0], _O_E_STAR)
    assert float(w[0]) > 0.7
    res = apply_ridging(a_cat, h_cat, Z, Z, _f([closing]), n_cat, dt,
                        e_star=_O_E_STAR, mu_rdg=_O_MU_RDG, H_star=_O_H_STAR,
                        snow_fraction_retained=_O_SNOW_RETAINED)
    # cat0 fully consumed => its pre-clip draw exceeded a_cat0 (the cap bound);
    # H_min = 2 h_part > h0 so no ridge returns to cat0.
    assert float(res["a"][0][0]) < 1e-9
    dA = float(jnp.sum(res["a"]) - jnp.sum(a_cat))
    assert abs(dA) < 0.9 * closing * dt                       # net closing under-delivers
    assert dA < 0.0                                           # but still net-convergent
    V_old = float(jnp.sum(h_cat * a_cat))
    V_new = float(jnp.sum(res["h"] * res["a"]))
    assert V_new == pytest.approx(V_old, rel=1e-11, abs=0.0)  # volume still conserved
    assert bool(jnp.all(jnp.isfinite(res["a"])))


def test_ridging_conserves_salt_mass():
    (_, a_cat, h_cat, _v, S_ice, _p), res = _apply(0.5e-6)
    salt_old = float(jnp.sum(S_ice * h_cat * a_cat)) * float(constants.rho_ice) * _O_PSU_TO_FRACTION
    salt_new = float(jnp.sum(res["S_ice"] * res["h"] * res["a"])) * float(constants.rho_ice) * _O_PSU_TO_FRACTION
    assert salt_new == pytest.approx(salt_old, rel=1e-11, abs=0.0)


def test_ridging_snow_retained_fraction_and_ocean_deficit():
    """Single-donor expected-value pin: the ridge retains exactly
    snow_fraction_retained of the donated snow, and the rest is reported to the
    ocean — (V_snow_old - V_snow_new) rho_snow == snow_to_ocean * dt."""
    n_cat = 5
    h0, a0, Vs0 = 0.5, 0.6, 0.05
    a_cat = _f([[a0, 0, 0, 0, 0]])
    h_cat = _f([[h0, 0, 0, 0, 0]])
    V_snow = _f([[Vs0, 0, 0, 0, 0]])
    Z = jnp.zeros((1, n_cat))
    dt = 3600.0
    res = apply_ridging(a_cat, h_cat, V_snow, Z, _f([0.2e-6]), n_cat, dt,
                        e_star=_O_E_STAR, mu_rdg=_O_MU_RDG, H_star=_O_H_STAR,
                        snow_fraction_retained=_O_SNOW_RETAINED)
    da0 = float(a0 - res["a"][0][0])                          # donor area removed
    Vsnow_donated = Vs0 * (da0 / a0)                          # snow rides with area fraction
    vs_new = float(jnp.sum(res["V_snow"]))
    snow_ocean = float(res["snow_to_ocean"][0])
    # retained: new snow == donor remainder + retained fraction of the donation
    assert vs_new == pytest.approx(
        (Vs0 - Vsnow_donated) + _O_SNOW_RETAINED * Vsnow_donated, rel=1e-9, abs=0.0)
    # shed: reported ocean flux == the un-retained donation
    assert snow_ocean * dt == pytest.approx(
        (1.0 - _O_SNOW_RETAINED) * Vsnow_donated * float(constants.rho_snow),
        rel=1e-9, abs=0.0)
    assert Vsnow_donated > 0.0


def test_ridging_pond_fully_drains_expected_value():
    """Single-donor expected-value pin: ALL pond water on ridging ice drains to
    the ocean (ridges carry no pond) — drained == pond-fraction of the donation,
    and (V_pond_old - V_pond_new) rho_water == pond_to_ocean * dt."""
    n_cat = 5
    h0, a0, Vp0 = 0.5, 0.6, 0.02
    a_cat = _f([[a0, 0, 0, 0, 0]])
    h_cat = _f([[h0, 0, 0, 0, 0]])
    V_pond = _f([[Vp0, 0, 0, 0, 0]])
    Z = jnp.zeros((1, n_cat))
    dt = 3600.0
    res = apply_ridging(a_cat, h_cat, Z, Z, _f([0.2e-6]), n_cat, dt,
                        V_pond_cat=V_pond, e_star=_O_E_STAR, mu_rdg=_O_MU_RDG,
                        H_star=_O_H_STAR, snow_fraction_retained=_O_SNOW_RETAINED)
    da0 = float(a0 - res["a"][0][0])
    drained_expected = Vp0 * (da0 / a0)
    drained = float(jnp.sum(V_pond) - jnp.sum(res["V_pond"]))
    pond_ocean = float(res["pond_to_ocean"][0])
    assert drained == pytest.approx(drained_expected, rel=1e-9, abs=0.0)
    assert pond_ocean * dt == pytest.approx(
        drained_expected * float(constants.rho_water), rel=1e-9, abs=0.0)
    assert drained > 0.0


def test_ridging_no_op_when_divergent():
    """A divergent column (closing_rate <= 0) does not ridge: EVERY returned state
    field is unchanged and both ocean fluxes are zero."""
    (_, a_cat, h_cat, V_snow, S_ice, V_pond), res = _apply(-1.0e-6)
    assert bool(jnp.allclose(res["a"], a_cat))
    assert bool(jnp.allclose(res["h"], h_cat))
    assert bool(jnp.allclose(res["V_snow"], V_snow))
    assert bool(jnp.allclose(res["S_ice"], S_ice))
    assert bool(jnp.allclose(res["V_pond"], V_pond))
    assert float(res["snow_to_ocean"][0]) == 0.0
    assert float(res["pond_to_ocean"][0]) == 0.0


# --- Hibler / Lipscomb ridge-thickness range (eq. 26) --------------------------

def _hibler_range(h_part, mu, H_star, hi_top):
    """Independent eq. 26 range with the production regularizations."""
    H_min = 2.0 * h_part
    H_max = min(mu * math.sqrt(h_part), H_star)
    H_max = min(H_max, hi_top)
    H_max_normal = max(H_max, H_min + 1.0e-3)
    if H_min > hi_top - 1.0e-3:                               # over-thick collapse
        return hi_top - 1.0e-3, hi_top
    return H_min, H_max_normal


def test_ridge_thickness_range_per_bin_overlap():
    """With a single thin donor (h_part = h0 analytic), the ridged ice is
    distributed UNIFORMLY in h over [H_min, H_max]: the per-receiver-bin area and
    volume match the independent overlap integral (pins the distribution, not
    just its support and first moment)."""
    n_cat = 5
    h0 = 0.5
    a0 = 0.6
    a_cat = _f([[a0, 0, 0, 0, 0]])
    h_cat = _f([[h0, 0, 0, 0, 0]])
    Z = jnp.zeros((1, n_cat))
    res = apply_ridging(a_cat, h_cat, Z, Z, _f([0.2e-6]), n_cat, 3600.0,
                        e_star=_O_E_STAR, mu_rdg=_O_MU_RDG, H_star=_O_H_STAR,
                        snow_fraction_retained=_O_SNOW_RETAINED)
    lo, hi = category_bounds(n_cat), upper_bounds(n_cat)
    a_new, h_new = res["a"][0], res["h"][0]
    da0 = float(a0 - a_new[0])
    V_ridge = da0 * h0
    H_min, H_max = _hibler_range(h0, _O_MU_RDG, _O_H_STAR, float(hi[-1]))
    H_width = H_max - H_min
    H_mean = 0.5 * (H_min + H_max)
    a_ridge = V_ridge / H_mean
    # per-bin uniform-g overlap
    for j in range(1, n_cat):                                # cat0 cannot receive (H_min>hi[0])
        a_over = max(float(lo[j]), H_min)
        b_over = min(float(hi[j]), H_max)
        overlap = max(b_over - a_over, 0.0)
        area_j = a_ridge * overlap / H_width
        vol_j = area_j * (0.5 * (a_over + b_over) if overlap > 0 else 0.0)
        assert float(a_new[j]) == pytest.approx(area_j, rel=1e-9, abs=1e-15)
        assert float(a_new[j] * h_new[j]) == pytest.approx(vol_j, rel=1e-9, abs=1e-15)


def test_ridge_thickness_range_h_star_binds():
    """H_star caps H_max: with h0=1 (mu sqrt(1)=4) and H_star=3, H_max=3, so the
    ridged mean thickness is (H_min+H_max)/2 = (2+3)/2 = 2.5 (a wrong/ignored
    H_star would not give 2.5)."""
    n_cat = 5
    h0, a0, H_star = 1.0, 0.6, 3.0
    a_cat = _f([[a0, 0, 0, 0, 0]])
    h_cat = _f([[h0, 0, 0, 0, 0]])
    Z = jnp.zeros((1, n_cat))
    res = apply_ridging(a_cat, h_cat, Z, Z, _f([0.2e-6]), n_cat, 3600.0,
                        e_star=_O_E_STAR, mu_rdg=_O_MU_RDG, H_star=H_star,
                        snow_fraction_retained=_O_SNOW_RETAINED)
    a_new, h_new = res["a"][0], res["h"][0]
    gained_area = float(jnp.sum(a_new[1:]))
    gained_vol = float(jnp.sum(a_new[1:] * h_new[1:]))
    assert gained_vol / gained_area == pytest.approx(2.5, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("h0,label", [(9.0, "H_max-floor (h>(mu/2)^2)"), (60.0, "over-thick collapse")])
def test_ridge_thickness_regularizations_conserve_volume(h0, label):
    """The two eq. 26 regularizations stay finite and volume-conserving: the
    H_max floor (mu sqrt(h_part) < 2 h_part, i.e. h_part>4) and the over-thick
    top-bin collapse (H_min > hi[-1])."""
    n_cat = 5
    a0 = 0.3
    a_cat = _f([[0, 0, 0, 0, a0]])                            # all ice in the top category
    h_cat = _f([[0, 0, 0, 0, h0]])
    Z = jnp.zeros((1, n_cat))
    res = apply_ridging(a_cat, h_cat, Z, Z, _f([0.2e-6]), n_cat, 3600.0,
                        e_star=_O_E_STAR, mu_rdg=_O_MU_RDG, H_star=_O_H_STAR,
                        snow_fraction_retained=_O_SNOW_RETAINED)
    V_old = float(jnp.sum(h_cat * a_cat))
    V_new = float(jnp.sum(res["h"] * res["a"]))
    assert V_new == pytest.approx(V_old, rel=1e-11, abs=0.0)
    assert bool(jnp.all(jnp.isfinite(res["a"]))) and bool(jnp.all(jnp.isfinite(res["h"])))


# --- config / default equivalence + constant canaries --------------------------

def test_ridging_defaults_equal_explicit_config():
    """apply_ridging with NO closure kwargs uses the RidgingConfig defaults —
    identical to passing them explicitly (pins the effective production defaults,
    not just the config literals).  Uses NONZERO snow so an altered
    snow_fraction_retained default would change V_snow / snow_to_ocean and fail."""
    a = _f([[0.3, 0.25, 0.2, 0.1, 0.05]])
    h = _f([[0.2, 0.6, 1.5, 3.0, 6.0]])
    V_snow = _f([[0.03, 0.02, 0.01, 0.005, 0.002]])
    S_ice = _f([[5.0, 5.0, 4.0, 3.0, 2.0]])
    r_def = apply_ridging(a, h, V_snow, S_ice, _f([0.5e-6]), 5, 3600.0)
    r_exp = apply_ridging(a, h, V_snow, S_ice, _f([0.5e-6]), 5, 3600.0,
                          e_star=_O_E_STAR, mu_rdg=_O_MU_RDG, H_star=_O_H_STAR,
                          snow_fraction_retained=_O_SNOW_RETAINED)
    for k in ("a", "h", "V_snow", "S_ice", "snow_to_ocean"):
        assert bool(jnp.allclose(r_def[k], r_exp[k]))


def test_ridging_constants_match_config():
    cfg = RidgingConfig()
    assert cfg.e_star == _O_E_STAR == 0.36
    assert cfg.mu_rdg == _O_MU_RDG == 4.0
    assert cfg.H_star == _O_H_STAR == 100.0
    assert cfg.snow_fraction_retained == _O_SNOW_RETAINED == 0.5
    assert _PSU_TO_FRACTION == _O_PSU_TO_FRACTION == 1.0e-3


# --- AD-safety -----------------------------------------------------------------

def test_ridging_grad_finite_x64_and_float32():
    """grad of the shed-snow flux wrt the closing rate (through the aksum
    compression and the where/max guards) is finite and positive (more
    convergence -> more ridging -> more snow shed) in x64 and float32, with the
    correct output dtype."""
    n_cat, a_cat, h_cat, V_snow, S_ice, V_pond = _column()

    def _snow_flux(closing_scalar, cast):
        res = apply_ridging(cast(a_cat), cast(h_cat), cast(V_snow), cast(S_ice),
                            closing_scalar.reshape((1,)), n_cat, 3600.0,
                            V_pond_cat=cast(V_pond), e_star=_O_E_STAR,
                            mu_rdg=_O_MU_RDG, H_star=_O_H_STAR,
                            snow_fraction_retained=_O_SNOW_RETAINED)
        return jnp.sum(res["snow_to_ocean"])

    def _check(cast, c0, want_dtype):
        g = jax.grad(lambda c: _snow_flux(c, cast))(c0)
        assert bool(jnp.isfinite(g)) and float(g) > 0.0
        assert g.dtype == want_dtype

    _check(lambda x: x.astype(jnp.float64), _f(0.5e-6), jnp.float64)
    # Drop x64 AND clear the lru_cached category bounds so the float32 pass does
    # not inherit float64 bounds that would promote the computation.
    jax.config.update("jax_enable_x64", False)
    category_bounds.cache_clear()   # upper_bounds delegates to it, no own cache
    _check(lambda x: x.astype(jnp.float32), jnp.array(0.5e-6, dtype=jnp.float32), jnp.float32)
    category_bounds.cache_clear()                             # restore clean cache state
    # autouse fixture restores the entry x64 state afterwards
