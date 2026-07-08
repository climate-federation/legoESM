"""Unit tests for the solar-induced fluorescence (SIF) diagnostic.

Exercises the leaf module ``legoesm.land.canopy.sif`` directly (per-function),
plus the physical-realism gate and the param-spec sanity that the CI ratchet
relies on.  The two wiring paths (two-leaf canopy, SimpleSEB big-leaf) are
exercised for realism in ``tests/land/integration`` / the realism driver; here
we test the pure SIF numerics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.land.canopy.sif import (
    __param_spec__,
    SIFConfig,
    actual_electron_transport,
    degree_of_light_saturation,
    fluorescence_yield,
    leaf_sif,
    leaf_sif_from_je,
    multilayer_canopy_sif,
    two_leaf_canopy_sif,
)


CFG = SIFConfig()


# --- fluorescence yield --------------------------------------------------

def test_yield_bounded_and_nonmonotonic_in_x():
    x = jnp.linspace(0.0, 1.0, 21)
    fs = fluorescence_yield(x, CFG)
    # Physically a yield fraction in (0, 1), and for a real C3 leaf it stays in
    # a narrow ~1-1.5% band across the light-saturation range.
    assert jnp.all(fs > 0.0) and jnp.all(fs < 0.1)
    # The van der Tol (2014) steady-state yield is NON-monotonic in x: it rises
    # as photochemistry saturates, dips as NPQ (Kn) engages, then rises again as
    # the photochemical drain term (1-x) vanishes. Assert that shape rather than
    # a false monotonicity: an interior peak below x~0.5 (rise then fall).
    fs_np = fs
    assert float(fs_np[3]) > float(fs_np[0])          # initial rise off the dark limit
    peak_idx = int(jnp.argmax(fs_np))
    assert 0 < peak_idx < len(fs_np) - 1              # peak is interior, not an endpoint
    assert float(fs_np[peak_idx]) > float(fs_np[-1])  # dips after the NPQ peak


def test_yield_physically_realistic_midday():
    # Typical midday C3 leaf: yield fs ~ 1-2% (emitted SIF / APAR photon ratio).
    x = jnp.asarray(0.42)
    fs = float(fluorescence_yield(x, CFG))
    assert 0.005 < fs < 0.06, fs


def test_yield_endpoints_closed_form():
    # x=1 (fully saturated): ps=0, Kn=kn0 -> fs = kf/(kf+kd+kn0).
    fs1 = float(fluorescence_yield(jnp.asarray(1.0), CFG))
    assert fs1 == pytest.approx(CFG.kf / (CFG.kf + CFG.kd + CFG.kn0), rel=1e-6)


# --- electron transport & light saturation -------------------------------

def test_je_nonnegative_and_zero_below_compensation():
    An = jnp.asarray([15.0, 0.0])
    Ci = jnp.asarray([280.0, 40.0])       # 2nd: Ci < Gamma*
    gstar = jnp.asarray([43.0, 43.0])
    je = actual_electron_transport(An, Ci, gstar)
    assert jnp.all(je >= 0.0)
    assert float(je[1]) == 0.0            # An=0 below compensation -> je=0


def test_je_sign_guard_below_compensation_with_positive_an():
    # Regression (codex HIGH): Ci < Gamma* makes the true denominator negative,
    # so signed je < 0 for An > 0 and MUST floor to 0 (BEPS `if je<0: je=0`).
    # A one-sided max(denom, eps) would flip the sign -> huge positive je -> x=0.
    An = jnp.asarray(12.0)
    Ci = jnp.asarray(30.0)                 # < Gamma*
    gstar = jnp.asarray(43.0)
    je = actual_electron_transport(An, Ci, gstar)
    assert float(je) == 0.0
    # -> degree of light saturation is 1 (leaf at/below compensation), NOT 0.
    x = degree_of_light_saturation(je, jnp.asarray(600.0), CFG)
    assert float(x) == 1.0


def test_je_finite_gradient_at_compensation_point():
    # Ci == Gamma* is the guarded 0/0; gradient wrt An and Ci must stay finite.
    g = jax.grad(lambda An, Ci: actual_electron_transport(An, Ci, jnp.asarray(43.0)),
                 argnums=(0, 1))(jnp.asarray(5.0), jnp.asarray(43.0))
    assert all(jnp.isfinite(gi) for gi in g)


def test_light_saturation_in_unit_interval_and_dark_zero():
    je = jnp.asarray([23.0, 5.0, 0.0])
    apar = jnp.asarray([800.0, 40.0, 0.0])   # last: dark
    x = degree_of_light_saturation(je, apar, CFG)
    assert jnp.all(x >= 0.0) and jnp.all(x <= 1.0)
    assert float(x[2]) == 0.0                # dark -> x=0 (no 0/0)


# --- leaf SIF ------------------------------------------------------------

def test_leaf_sif_zero_in_dark_positive_in_light():
    An = jnp.asarray(15.0)
    Ci = jnp.asarray(280.0)
    gstar = jnp.asarray(43.0)
    assert float(leaf_sif(An, Ci, gstar, jnp.asarray(0.0), CFG)) == 0.0
    assert float(leaf_sif(An, Ci, gstar, jnp.asarray(800.0), CFG)) > 0.0


def test_leaf_sif_scales_with_apar_at_fixed_saturation():
    # Hold the degree of light saturation x fixed by scaling An with APAR so
    # je/APAR is constant; then SIF should scale ~linearly with APAR.
    gstar = jnp.asarray(43.0)
    Ci = jnp.asarray(280.0)
    # je = An*(Ci+2g*)/(Ci-g*); pick An so that x is the same for both APAR.
    factor = float((Ci + 2 * gstar) / (Ci - gstar))
    apar1, apar2 = 400.0, 800.0
    # je1/apar1 == je2/apar2  =>  An scales with APAR
    An1 = jnp.asarray(0.02 * apar1 / factor)
    An2 = jnp.asarray(0.02 * apar2 / factor)
    s1 = float(leaf_sif(An1, Ci, gstar, jnp.asarray(apar1), CFG))
    s2 = float(leaf_sif(An2, Ci, gstar, jnp.asarray(apar2), CFG))
    assert s2 == pytest.approx(2.0 * s1, rel=1e-6)


def test_leaf_sif_yield_fraction_realistic():
    An, Ci, gstar, apar = (
        jnp.asarray(15.0), jnp.asarray(280.0), jnp.asarray(43.0), jnp.asarray(800.0))
    sif = float(leaf_sif(An, Ci, gstar, apar, CFG))
    assert 0.005 < sif / float(apar) < 0.06     # emitted yield 0.5-6%


# --- canopy aggregation & escape probability -----------------------------

def test_two_leaf_sum_and_fesc():
    args = dict(
        An_sun=jnp.asarray(18.0), Ci_sun=jnp.asarray(280.0),
        gstar_sun=jnp.asarray(45.0), apar_sun=jnp.asarray(900.0),
        An_sh=jnp.asarray(6.0), Ci_sh=jnp.asarray(300.0),
        gstar_sh=jnp.asarray(43.0), apar_sh=jnp.asarray(200.0),
    )
    total = float(two_leaf_canopy_sif(cfg=CFG, **args))
    sun = float(leaf_sif(args["An_sun"], args["Ci_sun"], args["gstar_sun"], args["apar_sun"], CFG))
    sh = float(leaf_sif(args["An_sh"], args["Ci_sh"], args["gstar_sh"], args["apar_sh"], CFG))
    assert total == pytest.approx(sun + sh, rel=1e-6)   # fesc=1 default

    cfg_half = CFG._replace(escape_probability=0.5)
    total_half = float(two_leaf_canopy_sif(cfg=cfg_half, **args))
    assert total_half == pytest.approx(0.5 * total, rel=1e-6)


# --- multilayer canopy (CLM-ML) aggregation ------------------------------

def _je(An, Ci, gstar):  # big-leaf-style inversion, for bridging to leaf_sif
    return actual_electron_transport(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gstar))


def test_leaf_sif_from_je_matches_leaf_sif():
    # leaf_sif is the An/Ci wrapper around leaf_sif_from_je(actual_electron_transport(...)).
    An, Ci, gs, ap = 15.0, 280.0, 43.0, 800.0
    a = float(leaf_sif(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gs), jnp.asarray(ap), CFG))
    b = float(leaf_sif_from_je(_je(An, Ci, gs), jnp.asarray(ap), CFG))
    assert a == pytest.approx(b, rel=1e-12)


def test_multilayer_reduces_to_leaf_sif_single_element():
    # One element, leaf_area=1; feeding je = actual_electron_transport(An,Ci,Γ*)
    # reproduces the big-leaf leaf_sif × fesc — the cross-check tying the
    # multilayer (je-native) path to the big-leaf/two-leaf core.
    An, Ci, gs, ap = 15.0, 280.0, 43.0, 800.0
    je = jnp.asarray([[_je(An, Ci, gs)]]); apar = jnp.asarray([[ap]]); la = jnp.asarray([[1.0]])
    ml = float(multilayer_canopy_sif(je, apar, la, CFG)[0])
    ref = float(leaf_sif(jnp.asarray(An), jnp.asarray(Ci), jnp.asarray(gs), jnp.asarray(ap), CFG))
    assert ml == pytest.approx(ref, rel=1e-6)


def test_multilayer_two_elements_equals_two_leaf():
    # Two elements (sunlit, shaded) with leaf_area=1 must equal two_leaf_canopy_sif
    # of the same two classes (bridged via je inversion) — same fluorescence core.
    je = jnp.asarray([[_je(18.0, 280.0, 45.0), _je(6.0, 300.0, 43.0)]])
    apar = jnp.asarray([[900.0, 200.0]]); la = jnp.asarray([[1.0, 1.0]])
    ml = float(multilayer_canopy_sif(je, apar, la, CFG)[0])
    tl = float(two_leaf_canopy_sif(
        jnp.asarray(18.0), jnp.asarray(280.0), jnp.asarray(45.0), jnp.asarray(900.0),
        jnp.asarray(6.0), jnp.asarray(300.0), jnp.asarray(43.0), jnp.asarray(200.0), CFG))
    assert ml == pytest.approx(tl, rel=1e-6)


def test_multilayer_leaf_area_weighting_and_zero_mask():
    # leaf_area scales the per-ground contribution linearly; a zero-area element
    # (unfilled layer) drops out entirely.
    je = jnp.asarray([[30.0, 30.0]]); apar = jnp.asarray([[800.0, 800.0]])
    one = float(multilayer_canopy_sif(je, apar, jnp.asarray([[1.0, 0.0]]), CFG)[0])
    two = float(multilayer_canopy_sif(je, apar, jnp.asarray([[2.0, 0.0]]), CFG)[0])
    both = float(multilayer_canopy_sif(je, apar, jnp.asarray([[1.0, 1.0]]), CFG)[0])
    assert two == pytest.approx(2.0 * one, rel=1e-6)     # linear in leaf area
    assert both == pytest.approx(2.0 * one, rel=1e-6)     # the zeroed element is inert


def test_multilayer_per_column_reduction():
    # (ncol, K) -> (ncol,) independent per column.
    je = jnp.asarray([[30.0, 12.0], [30.0, 12.0]])
    apar = jnp.asarray([[800.0, 200.0], [0.0, 0.0]])       # col 1 dark
    la = jnp.asarray([[1.0, 1.0], [1.0, 1.0]])
    out = multilayer_canopy_sif(je, apar, la, CFG)
    assert out.shape == (2,)
    assert float(out[0]) > 0.0 and float(out[1]) == 0.0


# --- differentiability ---------------------------------------------------

def test_finite_gradients_wrt_An_and_APAR():
    def f(An, apar):
        return leaf_sif(An, jnp.asarray(280.0), jnp.asarray(43.0), apar, CFG)
    gAn, gAp = jax.grad(f, argnums=(0, 1))(jnp.asarray(15.0), jnp.asarray(800.0))
    assert jnp.isfinite(gAn) and jnp.isfinite(gAp)
    # Gradient must also be finite at the dark edge (APAR -> 0, the guarded 0/0).
    gAn0, gAp0 = jax.grad(f, argnums=(0, 1))(jnp.asarray(0.0), jnp.asarray(0.0))
    assert jnp.isfinite(gAn0) and jnp.isfinite(gAp0)


def test_finite_gradient_wrt_kn_gamma_at_light_saturation():
    # kn_gamma is a tunable param; x = degree_of_light_saturation hits exactly 0
    # at light saturation (je >= max_electron_yield*apar), where power(0,gamma)
    # has a 0*log(0) = NaN gradient wrt the exponent unless guarded.  grad wrt
    # kn_gamma must stay finite there (and at x in (0,1)).
    def sif_of_gamma(kg, je, apar):
        cfg = CFG._replace(kn_gamma=kg)
        return leaf_sif_from_je(jnp.asarray(je), jnp.asarray(apar), cfg)
    # Saturated: je=100 >> 0.05*100=5 -> x clipped to 0.
    g_sat = jax.grad(sif_of_gamma)(jnp.asarray(2.83), 100.0, 100.0)
    # Interior: x in (0,1).
    g_mid = jax.grad(sif_of_gamma)(jnp.asarray(2.83), 20.0, 800.0)
    assert jnp.isfinite(g_sat) and jnp.isfinite(g_mid)
    assert float(g_sat) == 0.0  # analytic limit: d/dgamma [0^gamma] = 0


# --- param-spec / config sanity ------------------------------------------

def test_param_spec_bounds_contain_defaults():
    spec = __param_spec__["SIFConfig"]["params"]
    defaults = SIFConfig()._asdict()
    for field, meta in spec.items():
        lo, hi = meta["bounds"]
        assert lo <= defaults[field] <= hi, (field, defaults[field], meta["bounds"])


def test_init_reexport():
    from legoesm.land.canopy import SIFConfig as SC
    assert SC is SIFConfig
