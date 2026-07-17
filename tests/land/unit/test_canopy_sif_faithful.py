"""Oracle-faithfulness pins for the canopy SIF diagnostic.

Oracle: the **BEPS-SIF** ``photosyn_gs.c`` fluorescence block (Qiu & Zhang,
``yongguangzhang/BEPS-SIF-model``), which the legoESM
``legoesm.land.canopy.sif`` module ports.  The verbatim C source is::

    // --- SIF_y (fluorescence yield fs from degree-of-light-saturation xxn) ---
    kf = 0.05;  kd = 0.95;  kp = 4.0;
    ps_sif = kp/(kf+kp+kd)*(1.0-xxn);
    kk = exp(log(xxn)*2.83);                    // = xxn**2.83
    kn = 2.48*(1+0.114)*kk/(kk+0.114);          // van der Tol (2014) NPQ
    fm = kf/(kf+kd+kn);
    fs = fm*(1.0-ps_sif);
    // --- electron transport / light saturation / emitted flux ---
    je  = aphoto*(ci + 2.0*gammac)/(ci - gammac);   if (je<0)  je  = 0.0;
    xxn = 1.0 - je/(iphoton*0.05);                  if (xxn<0) xxn = 0.0;
    ffs = SIF_y(xxn);
    *xSIF = ffs*iphoton;                            // iphoton = 4.55*0.5*rad_leaf

The oracle below is an INDEPENDENT scalar (numpy) reimplementation with every
coefficient typed straight from that C source — never read back from the module
under test (no circular oracle).  ``fluorescence_yield``/``je``/``x`` and the
composed ``leaf_sif = fs(x)*APAR`` are pinned to round-off (rel 1e-12), and each
coefficient is canaried (perturb one -> the pin would fail) so it is provably
non-vacuous.

Complements ``test_canopy_sif.py`` (realism/property/aggregation); this file
owns the closed-form oracle match and the documented departures from BEPS-SIF:
absorbed-PAR (not incident-PPFD) reformulation, the ``Ci=Gamma*`` knife-edge
AD guard, the symmetric x-clip, and the additive ``fesc``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.land.canopy.sif import (
    SIFConfig,
    actual_electron_transport,
    degree_of_light_saturation,
    fluorescence_yield,
    leaf_sif,
    leaf_sif_from_je,
    two_leaf_canopy_sif,
)

CFG = SIFConfig()

# --- BEPS-SIF SIF_y oracle constants (photosyn_gs.c, Qiu 2015; verbatim) -----
# Typed from the C source, NOT from SIFConfig, so the pin is an independent
# check on the shipped defaults.
_O_KF = 0.05      # kf  fluorescence rate constant
_O_KD = 0.95      # kd  thermal / basal de-excitation
_O_KP = 4.0       # kp  photochemical rate constant
_O_KN0 = 2.48     # kn = kn0*(1+beta)*x^g/(x^g+beta)  (van der Tol 2014 NPQ)
_O_BETA = 0.114   # NOT 0.0114 (the drought param set) — SIF_y hardcodes 0.114
_O_GAMMA = 2.83   # kk = exp(log(xxn)*2.83) = xxn**2.83
_O_JE_GSTAR_FACTOR = 2.0   # je = aphoto*(ci + 2.0*gammac)/(ci - gammac)
_O_MEY = 0.05     # xxn = 1 - je/(iphoton*0.05); legoESM: max_electron_yield


def _fs_oracle(x, *, kf=_O_KF, kd=_O_KD, kp=_O_KP,
               kn0=_O_KN0, beta=_O_BETA, gamma=_O_GAMMA):
    """Independent scalar reimpl of BEPS-SIF ``SIF_y`` (fluorescence yield fs).

    Optional keyword overrides let a canary swap ONE coefficient to prove the
    pin is coefficient-sensitive.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    kk = x ** gamma
    kn = kn0 * (1.0 + beta) * kk / (kk + beta)
    fm = kf / (kf + kd + kn)
    ps = kp / (kf + kp + kd) * (1.0 - x)
    return fm * (1.0 - ps)


def _je_oracle(An, Ci, gstar):
    """BEPS ``je = aphoto*(ci+2*gammac)/(ci-gammac)`` floored at 0."""
    je = An * (Ci + _O_JE_GSTAR_FACTOR * gstar) / (Ci - gstar)
    return max(je, 0.0)


def _x_oracle(je, apar):
    """BEPS ``xxn = 1 - je/(apar*0.05)`` clipped ``[0,1]`` (upper<=1 since je>=0)."""
    xxn = 1.0 - je / (apar * _O_MEY)
    return min(max(xxn, 0.0), 1.0)


# --- shipped defaults are the BEPS-SIF values --------------------------------

def test_config_defaults_are_beps_coefficients():
    # The pin is only meaningful if the SHIPPED defaults equal the oracle
    # constants; catch a future default drift here (independent of the curves).
    assert CFG.kf == _O_KF
    assert CFG.kd == _O_KD
    assert CFG.kp == _O_KP
    assert CFG.kn0 == _O_KN0
    assert CFG.kn_beta == _O_BETA
    assert CFG.kn_gamma == _O_GAMMA
    assert CFG.max_electron_yield == _O_MEY
    assert CFG.escape_probability == 1.0   # fesc=1 -> raw BEPS emitted SIF


# --- fluorescence yield fs(x) matches SIF_y to round-off ---------------------

def test_fluorescence_yield_matches_SIF_y_curve():
    x = jnp.linspace(0.0, 1.0, 201)
    fs_mod = np.asarray(fluorescence_yield(x, CFG))
    np.testing.assert_allclose(fs_mod, _fs_oracle(np.asarray(x)), rtol=1e-12, atol=0.0)


def test_fluorescence_yield_clips_out_of_range_input():
    # legoESM in-domain guard: SIF_y is only defined on x in [0,1] (the C driver
    # clips xxn<0 BEFORE calling SIF_y and never feeds x>1), so fluorescence_yield
    # clips its INPUT to [0,1] to stay in-domain -- a robustness addition, NOT a
    # claim that SIF_y itself clips.  _fs_oracle clips the same way to match the
    # MODULE.
    assert float(fluorescence_yield(jnp.asarray(-0.3), CFG)) == pytest.approx(
        _fs_oracle(0.0), rel=1e-12)
    assert float(fluorescence_yield(jnp.asarray(1.7), CFG)) == pytest.approx(
        _fs_oracle(1.0), rel=1e-12)


@pytest.mark.parametrize(
    "param,wrong",
    [
        ("kn0", 3.0),
        ("beta", 0.0114),   # the factor-10 drought-set trap
        ("gamma", 2.0),
        ("kf", 0.06),
        ("kd", 1.05),
        ("kp", 5.0),
    ],
)
def test_fluorescence_yield_coefficient_canary(param, wrong):
    # The module matches the CORRECT oracle to round-off, and a single-coefficient
    # perturbation is clearly detectable -> the pin is non-vacuous (it is really
    # pinning THIS coefficient, not just any smooth curve).
    x = jnp.linspace(0.05, 0.95, 19)
    fs_mod = np.asarray(fluorescence_yield(x, CFG))
    np.testing.assert_allclose(fs_mod, _fs_oracle(np.asarray(x)), rtol=1e-12, atol=0.0)
    fs_wrong = _fs_oracle(np.asarray(x), **{param: wrong})
    assert np.max(np.abs(fs_mod - fs_wrong)) > 1e-4, (param, wrong)


# --- electron transport & light saturation match BEPS ------------------------

def test_actual_electron_transport_matches_beps_je():
    # Normal regime (Ci well above Gamma*): module == BEPS je exactly.
    An = jnp.asarray([5.0, 15.0, 25.0])
    Ci = jnp.asarray([250.0, 280.0, 400.0])
    gs = jnp.asarray([40.0, 43.0, 45.0])
    je_mod = np.asarray(actual_electron_transport(An, Ci, gs))
    je_ref = np.array([_je_oracle(float(a), float(c), float(g))
                       for a, c, g in zip(An, Ci, gs)])
    np.testing.assert_allclose(je_mod, je_ref, rtol=1e-12, atol=0.0)


def test_je_gstar_factor_canary():
    # The '2*Gamma*' numerator factor is load-bearing; dropping it to 1*Gamma*
    # shifts je measurably -> the pin is not vacuous on that coefficient.
    An, Ci, gs = 15.0, 280.0, 43.0
    je_mod = float(actual_electron_transport(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gs)))
    assert je_mod == pytest.approx(_je_oracle(An, Ci, gs), rel=1e-12)
    je_wrong = An * (Ci + 1.0 * gs) / (Ci - gs)
    assert abs(je_mod - je_wrong) > 1.0


def test_degree_of_light_saturation_matches_beps_xxn():
    je = jnp.asarray([0.0, 5.0, 23.0, 60.0])
    apar = jnp.asarray([800.0, 40.0, 800.0, 100.0])   # last: je>0.05*apar -> clip 0
    x_mod = np.asarray(degree_of_light_saturation(je, apar, CFG))
    x_ref = np.array([_x_oracle(float(j), float(a)) for j, a in zip(je, apar)])
    np.testing.assert_allclose(x_mod, x_ref, rtol=1e-12, atol=0.0)


def test_x_max_electron_yield_canary():
    je, apar = 20.0, 800.0
    x_mod = float(degree_of_light_saturation(jnp.asarray(je), jnp.asarray(apar), CFG))
    assert x_mod == pytest.approx(_x_oracle(je, apar), rel=1e-12)
    x_wrong = min(max(1.0 - je / (apar * 0.10), 0.0), 1.0)   # mey=0.10
    assert abs(x_mod - x_wrong) > 1e-3


# --- composed leaf SIF = fs(x) * APAR ----------------------------------------

def test_leaf_sif_matches_full_oracle_chain():
    An = jnp.asarray([5.0, 15.0, 25.0])
    Ci = jnp.asarray([250.0, 280.0, 400.0])
    gs = jnp.asarray([40.0, 43.0, 45.0])
    apar = jnp.asarray([300.0, 800.0, 1500.0])
    sif_mod = np.asarray(leaf_sif(An, Ci, gs, apar, CFG))
    sif_ref = np.array([
        _fs_oracle(_x_oracle(_je_oracle(float(a), float(c), float(g)), float(p))) * float(p)
        for a, c, g, p in zip(An, Ci, gs, apar)
    ])
    np.testing.assert_allclose(sif_mod, sif_ref, rtol=1e-12, atol=0.0)


def test_grad_leaf_sif_matches_oracle_finite_difference():
    # AD derivative wrt An reproduces the oracle chain's derivative (pins that the
    # differentiable path is the SAME function, not just finite).
    Ci, gs, apar = 280.0, 43.0, 800.0

    def f(An):
        return leaf_sif(An, jnp.asarray(Ci), jnp.asarray(gs), jnp.asarray(apar), CFG)

    An0 = 15.0
    g_ad = float(jax.grad(f)(jnp.asarray(An0)))
    h = 1e-4

    def chain(An):
        return _fs_oracle(_x_oracle(_je_oracle(An, Ci, gs), apar)) * apar

    g_fd = (chain(An0 + h) - chain(An0 - h)) / (2 * h)
    assert g_ad == pytest.approx(g_fd, rel=1e-6)


# --- documented departures from BEPS-SIF -------------------------------------

def test_departure_emitted_flux_uses_absorbed_par():
    # BEPS: *xSIF = SIF_y(xxn)*iphoton (INCIDENT PPFD). legoESM: SIF = fs*APAR
    # (ABSORBED PAR) — the emitted-flux multiplier is the absorbed-PAR argument,
    # the documented reformulation (retune max_electron_yield/fesc for absolute
    # SIF). Pin the multiplier structurally.
    je, apar = 25.0, 700.0
    x = degree_of_light_saturation(jnp.asarray(je), jnp.asarray(apar), CFG)
    fs = float(fluorescence_yield(x, CFG))
    sif = float(leaf_sif_from_je(jnp.asarray(je), jnp.asarray(apar), CFG))
    assert sif == pytest.approx(fs * apar, rel=1e-12)


def test_departure_compensation_knife_edge_takes_je_zero_branch():
    # Ci in (Gamma*, Gamma*+eps]: BEPS computes a huge je (tiny+ denom) then its
    # xxn<0 clip masks it to x=0; legoESM instead returns je=0 (-> x=1, since
    # APAR>eps here) for a finite gradient at Ci=Gamma*. Pin the je=0/x=1 branch,
    # and confirm the two branches are observably different so this is non-vacuous.
    gstar = 43.0
    Ci = gstar + 5.0e-10                       # within the module's eps of Gamma*
    apar = 800.0
    # Behavioral precondition (no private-symbol import): the guard puts je on the
    # je=0 branch at this knife-edge.
    assert float(actual_electron_transport(
        jnp.asarray(15.0), jnp.asarray(Ci), jnp.asarray(gstar))) == 0.0
    sif = float(leaf_sif(jnp.asarray(15.0), jnp.asarray(Ci), jnp.asarray(gstar),
                         jnp.asarray(apar), CFG))
    sif_je0 = _fs_oracle(1.0) * apar          # module branch: je=0 -> x=1
    sif_beps = _fs_oracle(0.0) * apar         # BEPS branch: huge je -> clip x=0
    assert sif == pytest.approx(sif_je0, rel=1e-12)
    assert abs(sif_je0 - sif_beps) > 1e-3     # branches differ -> canary is real


def test_departure_compensation_knife_edge_gradient_finite():
    # The whole point of the je=0 branch: reverse-mode grad wrt Ci stays finite
    # exactly at Ci=Gamma* (BEPS's C code has no gradient concern).
    gstar = 43.0

    def f(Ci):
        return leaf_sif(jnp.asarray(12.0), Ci, jnp.asarray(gstar),
                        jnp.asarray(800.0), CFG)

    g = float(jax.grad(f)(jnp.asarray(gstar)))
    assert np.isfinite(g)


def test_departure_below_compensation_negative_an_takes_je_zero():
    # An<0 AND Ci<Gamma* (respiring leaf below compensation): BEPS's je=neg/neg is
    # spuriously POSITIVE — its `if(je<0) je=0` floor misses the sign flip — giving
    # x=0; legoESM's denom>eps guard sends je->0 (x=1 for APAR>eps, as here), physical "no electron
    # transport at/below compensation". This is a WHOLE half-plane, NOT an eps set,
    # so it is a distinct departure from the positive-An knife-edge above.
    An, Ci, gstar, apar = -1.0, 40.0, 43.0, 800.0
    je_mod = float(actual_electron_transport(
        jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gstar)))
    assert je_mod == 0.0                                      # module: je=0 (x=1)
    je_beps = An * (Ci + _O_JE_GSTAR_FACTOR * gstar) / (Ci - gstar)
    assert je_beps > 0.0                                      # the BEPS artifact: je>0
    sif_mod = float(leaf_sif(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gstar),
                             jnp.asarray(apar), CFG))
    sif_je0 = _fs_oracle(1.0) * apar                         # module branch: x=1
    sif_beps = _fs_oracle(_x_oracle(je_beps, apar)) * apar    # BEPS branch: x=0
    assert sif_mod == pytest.approx(sif_je0, rel=1e-12)
    assert abs(sif_je0 - sif_beps) > 1e-3                    # branches differ -> real


def test_departure_low_positive_apar_dark_guard():
    # 0 < APAR <= eps: BEPS applies xxn=1-je/(APAR*0.05) (-> x=1 at je=0); legoESM's
    # APAR>eps dark guard returns x=0 for a finite gradient at APAR=0 (mirror of the
    # Ci=Gamma* guard). The YIELD branch differs from BEPS, but the emitted flux is
    # O(eps) so it is negligible in SIF.
    je, apar = 0.0, 5.0e-10
    x_mod = float(degree_of_light_saturation(jnp.asarray(je), jnp.asarray(apar), CFG))
    assert x_mod == 0.0                                       # module: dark (guarded)
    assert _x_oracle(je, apar) == 1.0                        # BEPS: x=1 at je=0
    sif_mod = float(leaf_sif_from_je(jnp.asarray(je), jnp.asarray(apar), CFG))
    assert abs(sif_mod) < 1e-9                               # emitted flux O(eps) ~ 0


def test_departure_exact_compensation_point_stays_finite():
    # At EXACT Ci=Gamma* BEPS divides 0/0 (An=0 -> NaN) or je/0 (An!=0 -> +/-inf);
    # the denom>eps guard returns the finite je=0 (x=1 for APAR>eps, as here)
    # regardless of An sign.
    gstar = 43.0
    for An in (0.0, 15.0):
        je = float(actual_electron_transport(
            jnp.asarray(An), jnp.asarray(gstar), jnp.asarray(gstar)))
        assert np.isfinite(je) and je == 0.0
        sif = float(leaf_sif(jnp.asarray(An), jnp.asarray(gstar), jnp.asarray(gstar),
                             jnp.asarray(800.0), CFG))
        assert np.isfinite(sif)
        assert sif == pytest.approx(_fs_oracle(1.0) * 800.0, rel=1e-12)   # x=1 branch


def test_departure_exact_zero_apar_stays_finite():
    # At EXACT APAR=0 BEPS computes je/0 (0/0 -> NaN once je is floored); the
    # APAR>eps guard returns the finite x=0 (SIF=0). Here Ci>Gamma* so je is floored
    # to 0 first, leaving the module NaN-free and SIF exactly 0.
    sif = float(leaf_sif(jnp.asarray(-1.0), jnp.asarray(280.0), jnp.asarray(43.0),
                         jnp.asarray(0.0), CFG))
    assert np.isfinite(sif) and sif == 0.0
    x = float(degree_of_light_saturation(jnp.asarray(0.0), jnp.asarray(0.0), CFG))
    assert x == 0.0


def test_native_je_api_upper_clip_bounds_negative_je():
    # leaf_sif_from_je accepts a caller-supplied native je WITHOUT flooring it (the
    # CLM-ML path), so a stray je<0 (with APAR>eps) gives x = 1 - je/(mey*apar) > 1;
    # legoESM's symmetric [0,1] clip bounds x to 1 (BEPS clips only xxn<0). Pin the
    # degree_of_light_saturation upper clip DIRECTLY -- fluorescence_yield ALSO clips
    # its input, so the composed SIF alone would not isolate this clip (it would pass
    # even if the x-clip were removed).
    je, apar = -1.0, 800.0
    x_raw = 1.0 - je / (apar * _O_MEY)                # = 1.025, exceeds 1
    assert x_raw > 1.0
    x_mod = float(degree_of_light_saturation(jnp.asarray(je), jnp.asarray(apar), CFG))
    assert x_mod == 1.0                                # upper clip fires here (not vacuous)
    sif = float(leaf_sif_from_je(jnp.asarray(je), jnp.asarray(apar), CFG))
    assert sif == pytest.approx(_fs_oracle(1.0) * apar, rel=1e-12)         # -> fs(1)*APAR


def test_departure_fesc_additive_and_clamped():
    # fesc (Yang & van der Tol 2016) is a legoESM addition absent from BEPS SIF_y.
    # fesc=1 (default) -> the raw, un-attenuated legoESM emitted SIF (equal to BEPS's
    # fs*iphoton form ONLY under the absorbed=incident identification of departure #1;
    # it does not by itself undo that reformulation). fesc in (0,1) scales linearly;
    # fesc is CLAMPED to [0,1] at use so a hand-set/perturbed value cannot produce
    # negative or amplified SIF.
    args = dict(
        An_sun=jnp.asarray(18.0), Ci_sun=jnp.asarray(280.0),
        gstar_sun=jnp.asarray(45.0), apar_sun=jnp.asarray(900.0),
        An_sh=jnp.asarray(6.0), Ci_sh=jnp.asarray(300.0),
        gstar_sh=jnp.asarray(43.0), apar_sh=jnp.asarray(200.0),
    )
    raw = float(two_leaf_canopy_sif(cfg=CFG, **args))
    sun = float(leaf_sif(args["An_sun"], args["Ci_sun"], args["gstar_sun"], args["apar_sun"], CFG))
    sh = float(leaf_sif(args["An_sh"], args["Ci_sh"], args["gstar_sh"], args["apar_sh"], CFG))
    assert raw == pytest.approx(sun + sh, rel=1e-12)          # fesc=1 -> un-attenuated
    scaled = float(two_leaf_canopy_sif(cfg=CFG._replace(escape_probability=0.7), **args))
    assert scaled == pytest.approx(0.7 * raw, rel=1e-12)      # linear in (0,1)
    lo = float(two_leaf_canopy_sif(cfg=CFG._replace(escape_probability=-0.2), **args))
    hi = float(two_leaf_canopy_sif(cfg=CFG._replace(escape_probability=1.2), **args))
    assert lo == 0.0                                          # clamp low -> 0
    assert hi == pytest.approx(raw, rel=1e-12)                # clamp high -> raw
