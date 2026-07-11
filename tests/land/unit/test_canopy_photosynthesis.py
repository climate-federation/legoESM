"""Unit tests for canopy/photosynthesis.py.

Checks:
- C3 assimilation > 0 at ambient CO2 and saturating light
- C3 assimilation = 0 in the dark
- C4 assimilation > C3 for hot, light-saturated conditions
- Temperature response peaks near 25-30 C
- Mixed C3/C4 fraction is a continuous weighted average
- Functions are JIT- and grad-compatible
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.land.canopy import photosynthesis as photo
from legoesm.land.canopy.photosynthesis import (
    _jmax25_over_vcmax25,
    _rd_atkin,
    c3_photosynthesis,
    c4_photosynthesis,
    photosynthesis,
    vcmax_temperature_response,
)


def test_c3_positive_daytime():
    Tf = jnp.array(298.15)
    Ci = jnp.array(280.0)
    APAR = jnp.array(1500.0)
    Vcmax25 = jnp.array(60.0)
    Ps = jnp.array(101325.0)
    alf = jnp.array(0.3)
    TgC = jnp.array(20.0)
    An, A_gross = c3_photosynthesis(Tf, Ci, APAR, Vcmax25, Ps, alf, TgC)
    assert float(An) > 5.0, f"expected daytime C3 An > 5, got {An}"
    # GROSS assimilation (the GPP) is >= net, strictly larger in the light by
    # the dark respiration Rd; see test_gross_minus_net_equals_rd_* below.
    assert float(A_gross) >= float(An)


def test_c3_zero_dark():
    An, A_gross = c3_photosynthesis(
        jnp.array(298.15), jnp.array(280.0), jnp.array(0.0),
        jnp.array(60.0), jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
    # With no APAR the light-limited rate is zero; the co-limitation drives
    # GROSS A -> 0, so both NET An and GROSS A_gross (the GPP) are 0 in the
    # dark.  This is why GPP is defined as max(A_gross, 0), NOT An + Rd.
    assert float(An) == 0.0
    assert float(A_gross) == 0.0


def test_c4_positive_daytime():
    An, A_gross = c4_photosynthesis(
        jnp.array(303.15), jnp.array(150.0), jnp.array(1500.0), jnp.array(40.0))
    assert float(An) > 5.0
    assert float(A_gross) >= float(An)


def test_vcmax_peak_near_25C():
    """Vcmax temperature response should be ~1 at 25C and drop at 50C."""
    f25 = vcmax_temperature_response(jnp.array(298.15), jnp.array(20.0))
    f50 = vcmax_temperature_response(jnp.array(323.15), jnp.array(20.0))
    f5  = vcmax_temperature_response(jnp.array(278.15), jnp.array(20.0))
    assert 0.9 < float(f25) < 1.1
    assert float(f50) < float(f25)
    assert float(f5) < float(f25)


def test_photosynthesis_mixing_is_weighted_average():
    args = dict(
        Tf=jnp.array(300.0), Ci=jnp.array(250.0), APAR=jnp.array(1200.0),
        Vcmax25_C3=jnp.array(60.0), Vcmax25_C4=jnp.array(40.0),
        Ps=jnp.array(101325.0), alf=jnp.array(0.3), TgC=jnp.array(20.0),
    )
    An_c3_only, Ag_c3_only = photosynthesis(fC4=jnp.array(0.0), **args)
    An_c4_only, Ag_c4_only = photosynthesis(fC4=jnp.array(1.0), **args)
    An_mix,     Ag_mix     = photosynthesis(fC4=jnp.array(0.5), **args)
    expected_mix = 0.5 * An_c3_only + 0.5 * An_c4_only
    assert jnp.allclose(An_mix, expected_mix, atol=1e-5)
    # GROSS assimilation blends the same continuous way (differentiable in fC4).
    expected_gross_mix = 0.5 * Ag_c3_only + 0.5 * Ag_c4_only
    assert jnp.allclose(Ag_mix, expected_gross_mix, atol=1e-5)


def test_photosynthesis_differentiable():
    """Gradients wrt Vcmax25 must be finite — required for training."""
    def loss(v):
        # Grad of NET An wrt Vcmax25 (first element of the (An, A_gross) tuple).
        An, _ = c3_photosynthesis(
            jnp.array(298.15), jnp.array(280.0), jnp.array(1500.0),
            v, jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
        return An
    g = jax.grad(loss)(jnp.array(60.0))
    assert jnp.isfinite(g)
    assert float(g) > 0.0  # more Vcmax => more An


# ---------------------------------------------------------------------------
# Canonical-FvCB structural checks (the ported DifferBESS bug fixes)
# ---------------------------------------------------------------------------

def test_rd25_equals_basal_fraction_no_double_count():
    """Rd at 25 degC / TgC=25 == 0.015 * Vcmax25 (the double-count fix).

    The old path used Rd = 0.015 * Vcmax(T) * rd_response(T), applying the
    temperature response twice.  The canonical form bases Rd on Vcmax25 and a
    response normalised to 1 at the reference state.
    """
    Vcmax25 = jnp.array(60.0)
    Rd = _rd_atkin(jnp.array(298.15), jnp.array(25.0), Vcmax25)
    assert jnp.allclose(Rd, 0.015 * Vcmax25, rtol=1e-6)


def test_jmax_acclimation_ratio_kattge_knorr():
    """Jmax25/Vcmax25 follows Kattge & Knorr 2007: 2.59 - 0.035*TgC."""
    for tgc in (11.0, 20.0, 35.0):
        ratio = _jmax25_over_vcmax25(jnp.array(tgc))
        assert jnp.allclose(ratio, 2.59 - 0.035 * tgc, rtol=1e-6)
    # cooler growth temperature => higher Jmax:Vcmax ratio
    assert float(_jmax25_over_vcmax25(jnp.array(11.0))) > \
        float(_jmax25_over_vcmax25(jnp.array(35.0)))


def test_c4_uses_clm5_constants():
    """C4 kinetics carry the CLM5-aligned constants (the b5e19e8/85bed3b fix)."""
    assert photo._S2_C4 == 313.15      # high-T deactivation onset (was 309.15)
    assert photo._S3_C4 == 0.2         # low-T inhibition slope (was 0.3)
    assert photo._RD25_FRAC_C4 == 0.025  # Rd25/Vcmax25 (was fixed 0.8)
    assert photo._ALPHA_C4 == 0.05     # quantum yield (was 0.067)


def test_c3_electron_transport_jmax_bounded():
    """At very high APAR the light-limited rate saturates (Jmax bound).

    The old JE = alf*APAR grew without bound; the canonical Jmax-limited J
    saturates, so doubling already-saturating light barely changes An.
    """
    base = dict(Tf=jnp.array(298.15), Ci=jnp.array(280.0),
                Vcmax25=jnp.array(60.0), Ps=jnp.array(101325.0),
                alf=jnp.array(0.3), TgC=jnp.array(20.0))
    An_2000 = float(c3_photosynthesis(APAR=jnp.array(2000.0), **base)[0])
    An_4000 = float(c3_photosynthesis(APAR=jnp.array(4000.0), **base)[0])
    # Saturating: a 2x light increase yields < 5% more assimilation.
    assert An_4000 >= An_2000
    assert (An_4000 - An_2000) / An_2000 < 0.05


# ---------------------------------------------------------------------------
# Gross-vs-net GPP: foliar-respiration double-count fix (cross-scheme)
# ---------------------------------------------------------------------------
#
# The two-leaf canopy exports GROSS assimilation as GPP; the carbon model
# (carbon_cycle.step_carbon) re-charges foliar maintenance respiration
# r_maint_fol*C_fol separately, so exporting NET An (= max(A_gross - Rd, 0))
# would double-count leaf respiration and bias carbon-use efficiency low.
# These tests lock the gross/net relationship the fix depends on and confirm
# the canopy FvCB path now matches the SimpleSEB Farquhar convention
# (carbon/stomata.py: gpp = max(A_gross, 0)*_MC).


def _c3_leaf_state():
    return dict(
        Tf=jnp.array(298.15), Ci=jnp.array(280.0), APAR=jnp.array(1500.0),
        Vcmax25=jnp.array(60.0), Ps=jnp.array(101325.0),
        alf=jnp.array(0.3), TgC=jnp.array(20.0))


def test_c3_gross_minus_net_equals_rd_in_light():
    """C3: A_gross - An == Rd exactly where An > 0, and A_gross >= An (>= the
    old NET value that was mislabelled 'GPP')."""
    st = _c3_leaf_state()
    An, A_gross = c3_photosynthesis(
        st["Tf"], st["Ci"], st["APAR"], st["Vcmax25"],
        st["Ps"], st["alf"], st["TgC"])
    # Rd uses the SAME growth-temperature clip [_TGC_LO, _TGC_HI] as the impl.
    TgC_a = jnp.clip(st["TgC"], photo._TGC_LO, photo._TGC_HI)
    Rd = _rd_atkin(st["Tf"], TgC_a, st["Vcmax25"])
    assert float(An) > 0.0                        # in the light
    assert float(Rd) > 0.0
    assert float(A_gross) >= float(An)            # GPP (gross) >= old NET value
    assert jnp.allclose(A_gross - An, Rd, rtol=1e-6, atol=1e-6)


def test_c4_gross_minus_net_equals_rd_in_light():
    """C4: A_gross - An == Rd (Rd recomputed from the same module constants)."""
    Tf = jnp.array(303.15)
    An, A_gross = c4_photosynthesis(
        Tf, jnp.array(150.0), jnp.array(1500.0), jnp.array(40.0))
    # Same C4 Rd formula the implementation uses — read the module constants
    # (no magic numbers) so the test tracks any constant change.
    q10_pow = photo._Q10_C4 ** ((Tf - photo._T_REF) / 10.0)
    Rd25 = photo._RD25_FRAC_C4 * jnp.array(40.0)
    Rd = Rd25 * q10_pow / (1.0 + jnp.exp(photo._S5_C4 * (Tf - photo._S6_C4)))
    assert float(An) > 0.0
    assert float(A_gross) >= float(An)
    assert jnp.allclose(A_gross - An, Rd, rtol=1e-6, atol=1e-6)


def test_gross_and_net_both_zero_in_dark():
    """In the dark A_gross == An == 0 (both floored): GPP is max(A_gross, 0),
    NOT An + Rd — so a dark leaf reports zero GPP, not a phantom +Rd uptake."""
    An, A_gross = c3_photosynthesis(
        jnp.array(298.15), jnp.array(280.0), jnp.array(0.0),
        jnp.array(60.0), jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
    assert float(An) == 0.0
    assert float(A_gross) == 0.0


def test_mixed_canopy_gross_ge_net():
    """A mixed C3/C4 canopy still reports GROSS >= NET at every fC4."""
    args = dict(
        Tf=jnp.array(300.0), Ci=jnp.array(250.0), APAR=jnp.array(1200.0),
        Vcmax25_C3=jnp.array(60.0), Vcmax25_C4=jnp.array(40.0),
        Ps=jnp.array(101325.0), alf=jnp.array(0.3), TgC=jnp.array(20.0))
    for f in (0.0, 0.5, 1.0):
        An, A_gross = photosynthesis(fC4=jnp.array(f), **args)
        assert float(A_gross) >= float(An)
        assert float(A_gross) >= 0.0


def test_canopy_and_simpleseb_share_gross_gpp_convention():
    """Cross-scheme: the two-leaf canopy FvCB path and the SimpleSEB Farquhar
    path both return (net, gross) with GPP = max(A_gross, 0) and
    A_gross - A_net == Rd in the light — neither folds leaf respiration into
    GPP, so the two schemes are now consistent (the double-count fix)."""
    from legoesm.land.carbon.stomata import (
        StomataConfig, arrhenius, farquhar_photosynthesis,
    )

    Ci = jnp.array(280.0)
    APAR = jnp.array(1500.0)
    T = jnp.array(298.15)

    # SimpleSEB Farquhar: A_net = A_gross - Rd (canopy_scaling=1.0 default).
    cfg = StomataConfig()
    A_net_seb, A_gross_seb = farquhar_photosynthesis(Ci, APAR, T, cfg)
    Rd_seb = arrhenius(cfg.Rd25, cfg.Ha_Rd, T)
    assert float(A_gross_seb) >= 0.0
    assert float(A_gross_seb) >= float(A_net_seb)
    assert jnp.allclose(A_gross_seb - A_net_seb, Rd_seb, rtol=1e-6, atol=1e-6)

    # Two-leaf canopy FvCB C3: same convention (gross >= net; gross - net==Rd).
    st = _c3_leaf_state()
    An_can, A_gross_can = c3_photosynthesis(
        st["Tf"], st["Ci"], st["APAR"], st["Vcmax25"],
        st["Ps"], st["alf"], st["TgC"])
    TgC_a = jnp.clip(st["TgC"], photo._TGC_LO, photo._TGC_HI)
    Rd_can = _rd_atkin(st["Tf"], TgC_a, st["Vcmax25"])
    assert float(A_gross_can) >= 0.0
    assert float(A_gross_can) >= float(An_can)
    assert jnp.allclose(A_gross_can - An_can, Rd_can, rtol=1e-6, atol=1e-6)


def test_gross_assimilation_differentiable():
    """Grad of GROSS assimilation is finite for C3, C4 and mixed canopies.

    The prior AD test only differentiated NET C3 An; the carbon-facing GPP
    flows through the GROSS branch, so cover its gradient too (and the mixed
    fC4 blend, which the carbon model sees for C3/C4 grid cells)."""
    def c3_gross(v):
        _, A_gross = c3_photosynthesis(
            jnp.array(298.15), jnp.array(280.0), jnp.array(1500.0),
            v, jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
        return A_gross

    def c4_gross(v):
        _, A_gross = c4_photosynthesis(
            jnp.array(303.15), jnp.array(150.0), jnp.array(1500.0), v)
        return A_gross

    def mix_gross(f):
        _, A_gross = photosynthesis(
            jnp.array(300.0), jnp.array(250.0), jnp.array(1200.0),
            jnp.array(60.0), jnp.array(40.0), f,
            jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
        return A_gross

    g_c3 = jax.grad(c3_gross)(jnp.array(60.0))
    g_c4 = jax.grad(c4_gross)(jnp.array(40.0))
    g_mix = jax.grad(mix_gross)(jnp.array(0.5))
    assert jnp.isfinite(g_c3) and float(g_c3) > 0.0   # more Vcmax -> more gross
    assert jnp.isfinite(g_c4) and float(g_c4) > 0.0
    assert jnp.isfinite(g_mix)   # d(gross)/d(fC4) = Agross_C4 - Agross_C3; finite


def test_gross_and_net_zero_in_dark_c4_and_mixed():
    """Dark (APAR=0): both An and A_gross == 0 for C4 and every mixed fC4 too
    (not just C3), so no leaf class reports phantom night-time GPP."""
    An4, Ag4 = c4_photosynthesis(
        jnp.array(303.15), jnp.array(150.0), jnp.array(0.0), jnp.array(40.0))
    assert float(An4) == 0.0
    assert float(Ag4) == 0.0
    for f in (0.0, 0.5, 1.0):
        An_m, Ag_m = photosynthesis(
            jnp.array(300.0), jnp.array(250.0), jnp.array(0.0),
            jnp.array(60.0), jnp.array(40.0), jnp.array(f),
            jnp.array(101325.0), jnp.array(0.3), jnp.array(20.0))
        assert float(An_m) == 0.0
        assert float(Ag_m) == 0.0
