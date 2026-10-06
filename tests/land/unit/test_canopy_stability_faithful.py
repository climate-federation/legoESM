"""Above-canopy MOST ORACLE-FAITHFULNESS tests.

``land/canopy/stability.py`` ports the CLM5 ``FrictionVelocityMod`` 4-regime
Monin-Obukhov similarity functions (Oleson et al. 2013 CLM5 Tech Note; Zeng et
al. 1998).  The existing ``test_canopy_stability.py`` is behavioral — it asserts
``ustar > 0`` / ``rah > 0`` / finite — it never pins the closed FORMS.

These pin the per-regime resistance forms to round-off (rel 1e-9) against an
INDEPENDENT scalar reimplementation of the CLM5 functions, at MATCHED
``zeta`` / ``z0`` / ``obu`` (the FORMS, not the solve — the iteration driver is a
documented departure, so a converged ``zeta`` is model-specific):

  * momentum ``ustar`` (:func:`_friction_velocity`) and heat ``ch``
    (:func:`_temperature_humidity_relation`) in all four regimes;
  * the neutral log-law limit and continuity across the free-convection matches;
  * the CLM5 INVERSE cube-root heat correction as the discriminator vs the
    gSAM/LSM4 sibling ``transfer_coef.f90`` (growing cube-root);
  * the ``kB^-1 = 0`` (``z0h = z0m``) departure vs the LSM4 ``kB^-1 = 2``;
  * the Zeng-1998 bulk-Richardson initialisation and the fixed-point convergence.

Independence: the gamma/beta/zeta_m/zeta_t/conv coefficients (and the stable
zeta-clamp bound) are local ``_O_*`` literals canaried against the module
constants (module == ``_O_*`` == value); the von Karman constant is canaried
against ``legoesm.constants``.  The guards EXERCISED here are the discarded-branch
``_MOST_ARG_FLOOR`` (eager ``where`` evaluates every regime, so an out-of-domain
log/cbrt would poison the VJP) and the stable ``zeta`` clamp (the Zeng-1998 init
drives it active).  The calm-wind (0.1, 1e-3 m/s) and resistance (1e-9) floors are
documented departures, not driven active by these inputs.
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
from legoesm.land.canopy.stability import (                           # noqa: E402
    _friction_velocity, _temperature_humidity_relation,
    _monin_obukhov_init, monin_obukhov_stability,
    compute_aerodynamics, compute_below_canopy_resistance,
    compute_boundary_layer_resistance,
    _ZETAM, _ZETAT, _MOST_BETA_STABLE, _MOST_GAMMA_UNSTABLE,
    _MOST_MOM_CONV_COEF, _MOST_HEAT_CONV_COEF, _RIB_MAX, _VIRT_T_COEF,
    _Z0MG_BARE, _NU_AIR, _CS_DENSE, _CS_BARE_COEF, _CS_BARE_EXP,
    _ZETA_MAX_STABLE,
)
from legoesm.land.canopy.config import CanopyConfig                  # noqa: E402

_ZW = CanopyConfig().zeta_cap_smoothing_width

# --- independent CLM5 FrictionVelocityMod oracle literals ----------------------
# (canaried in test_most_constants_match_clm5).
_O_KAPPA = 0.4     # von Karman
_O_GAMMA = 16.0    # Businger-Dyer (1 - 16 zeta)
_O_BETA = 5.0      # stable linear slope
_O_ZETAM = 1.574   # momentum very-unstable transition
_O_ZETAT = 0.465   # heat very-unstable transition
_O_MOM_CONV = 1.14    # momentum free-convection coefficient
_O_HEAT_CONV = 0.8    # heat free-convection coefficient
_O_RIB_MAX = 0.19  # Zeng-1998 bulk-Ri init cap
_O_ZETA_MAX_STABLE = 0.5  # stable-branch zeta clamp upper bound
# Beljaars & Holtslag (1991) stable-side coefficients (the DEPARTURE from CLM5's
# linear -5 zeta; b as rounded in the shared core implementation).
_O_BH_A, _O_BH_B, _O_BH_C, _O_BH_D = 1.0, 0.667, 5.0, 0.35


def _bh_psim(z):
    """Beljaars & Holtslag (1991) stable momentum psi (z >= 0)."""
    return -(_O_BH_A*z + _O_BH_B*(z - _O_BH_C/_O_BH_D)*math.exp(-_O_BH_D*z)
             + _O_BH_B*_O_BH_C/_O_BH_D)


def _bh_psih(z):
    """Beljaars & Holtslag (1991) stable heat psi (z >= 0)."""
    return -((1.0 + 2.0*_O_BH_A*z/3.0)**1.5
             + _O_BH_B*(z - _O_BH_C/_O_BH_D)*math.exp(-_O_BH_D*z)
             + _O_BH_B*_O_BH_C/_O_BH_D - 1.0)


def _psim(z):
    """Paulson (1970) unstable momentum psi (z <= 0)."""
    z = min(z, 0.0)
    x2 = math.sqrt(1.0 - _O_GAMMA * z)
    x = math.sqrt(x2)
    return 2.0*math.log((1.0+x)/2.0) + math.log((1.0+x2)/2.0) - 2.0*math.atan(x) + math.pi/2.0


def _psih(z):
    """Paulson unstable heat psi (z <= 0)."""
    z = min(z, 0.0)
    x2 = math.sqrt(1.0 - _O_GAMMA * z)
    return 2.0*math.log((1.0+x2)/2.0)


def _ustar_oracle(zldis, z0m, obu, um):
    """CLM5 FrictionVelocityMod friction velocity (unstable); BH91 stable side."""
    zeta = zldis/obu
    if zeta < -_O_ZETAM:                                    # very unstable
        d = (math.log(-_O_ZETAM*obu/z0m) - _psim(-_O_ZETAM) + _psim(z0m/obu)
             + _O_MOM_CONV*((-zeta)**(1.0/3.0) - _O_ZETAM**(1.0/3.0)))
    elif zeta < 0.0:                                        # unstable
        d = math.log(zldis/z0m) - _psim(zeta) + _psim(z0m/obu)
    else:                                                   # stable: BH91 (departure)
        d = math.log(zldis/z0m) - _bh_psim(zeta) + _bh_psim(z0m/obu)
    return _O_KAPPA*um/d


def _ch_oracle(zldis, z0h, obu):
    """CLM5 heat/scalar transfer (unstable); BH91 stable side (theta*/dtheta)."""
    zeta = zldis/obu
    if zeta < -_O_ZETAT:                                    # very unstable (INVERSE cbrt)
        d = (math.log(-_O_ZETAT*obu/z0h) - _psih(-_O_ZETAT) + _psih(z0h/obu)
             + _O_HEAT_CONV*(_O_ZETAT**(-1.0/3.0) - (-zeta)**(-1.0/3.0)))
    elif zeta < 0.0:                                        # unstable
        d = math.log(zldis/z0h) - _psih(zeta) + _psih(z0h/obu)
    else:                                                   # stable: BH91 (departure)
        d = math.log(zldis/z0h) - _bh_psih(zeta) + _bh_psih(z0h/obu)
    return _O_KAPPA/d


def _a(x):
    return jnp.array(float(x))


# regime -> obu that places zeta = zldis/obu (zldis=30) in that regime
_ZLDIS = 30.0
_REGIMES = [
    ("very-unstable", _ZLDIS/-3.0),
    ("unstable",      _ZLDIS/-0.5),
    ("stable",        _ZLDIS/+0.5),
    ("very-stable",   _ZLDIS/+3.0),
]


# --- per-regime form pins ------------------------------------------------------

@pytest.mark.parametrize("name,obu", _REGIMES)
def test_friction_velocity_matches_clm5_oracle(name, obu):
    """ustar matches the oracle form in every regime (CLM5 unstable, BH91 stable)."""
    z0m, um = 0.1, 4.0
    got = float(_friction_velocity(_a(_ZLDIS), _a(z0m), _a(obu), _a(um)))
    exp = _ustar_oracle(_ZLDIS, z0m, obu, um)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("name,obu", _REGIMES)
def test_heat_transfer_matches_clm5_oracle(name, obu):
    """ch matches the oracle heat form in every regime (CLM5 unstable, BH91 stable)."""
    z0h = 0.1
    got = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obu), _a(z0h)))
    exp = _ch_oracle(_ZLDIS, z0h, obu)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


def test_most_constants_match_clm5():
    """Module MOST constants equal the independent oracle literals and their
    documented CLM5 / Zeng-1998 values."""
    assert _MOST_GAMMA_UNSTABLE == _O_GAMMA == 16.0
    assert _MOST_BETA_STABLE == _O_BETA == 5.0
    assert _ZETAM == _O_ZETAM == 1.574
    assert _ZETAT == _O_ZETAT == 0.465
    assert _MOST_MOM_CONV_COEF == _O_MOM_CONV == 1.14
    assert _MOST_HEAT_CONV_COEF == _O_HEAT_CONV == 0.8
    assert _RIB_MAX == _O_RIB_MAX == 0.19
    assert _ZETA_MAX_STABLE == _O_ZETA_MAX_STABLE == 0.5
    assert float(constants.kappa_vk) == _O_KAPPA == 0.4


# --- defining-property / non-vacuity pins --------------------------------------

def test_neutral_limit_recovers_log_law():
    """As zeta -> 0 the forms collapse to the neutral log law
    ustar = kappa u / ln(z/z0), ch = kappa / ln(z/z0h)."""
    z0m, um = 0.1, 4.0
    obu = _ZLDIS/1e-6                                       # zeta ~ 1e-6 (near neutral)
    ustar = float(_friction_velocity(_a(_ZLDIS), _a(z0m), _a(obu), _a(um)))
    ch = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obu), _a(z0m)))
    assert ustar == pytest.approx(_O_KAPPA*um/math.log(_ZLDIS/z0m), rel=1e-4)
    assert ch == pytest.approx(_O_KAPPA/math.log(_ZLDIS/z0m), rel=1e-4)


def test_forms_continuous_across_free_convection_matches():
    """The very-unstable free-convection matches are C0: ustar (at zeta=-zetam)
    and ch (at zeta=-zetat) are continuous across the regime switch.  This
    catches a wrong match OFFSET (a mis-constructed free-convection branch would
    jump O(1) here); the free-convection COEFFICIENTS (1.14 / 0.8) themselves are
    pinned by the deep-regime form tests above (the correction is ~0 at the
    transition, so continuity alone cannot see them)."""
    z0m, um = 0.1, 4.0
    # momentum switch at zeta = -zetam
    obu_lo = _ZLDIS/(-_ZETAM - 1e-6)                       # very unstable
    obu_hi = _ZLDIS/(-_ZETAM + 1e-6)                       # unstable
    u_lo = float(_friction_velocity(_a(_ZLDIS), _a(z0m), _a(obu_lo), _a(um)))
    u_hi = float(_friction_velocity(_a(_ZLDIS), _a(z0m), _a(obu_hi), _a(um)))
    assert u_lo == pytest.approx(u_hi, rel=1e-5)
    # heat switch at zeta = -zetat
    obh_lo = _ZLDIS/(-_ZETAT - 1e-6)
    obh_hi = _ZLDIS/(-_ZETAT + 1e-6)
    c_lo = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obh_lo), _a(z0m)))
    c_hi = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obh_hi), _a(z0m)))
    assert c_lo == pytest.approx(c_hi, rel=1e-5)


def test_stable_branch_has_the_beljaars_holtslag_long_tail():
    """Stable side is BH91, not CLM5's linear -5 zeta: the resistance denom
    fm = kappa u / ustar grows with zeta but SLOWER than linear (the long tail),
    and starts with the Businger-Dyer slope ~5 near neutral."""
    z0m, um = 0.05, 4.0

    def fm(zeta):
        u = float(_friction_velocity(_a(_ZLDIS), _a(z0m), _a(_ZLDIS/zeta), _a(um)))
        return _O_KAPPA*um/u
    near = (fm(0.02) - fm(0.01)) / 0.01
    far = (fm(3.0) - fm(2.0)) / 1.0
    assert near == pytest.approx(_O_BETA, rel=0.05)
    assert 0.0 < far < 0.8*near


def test_very_unstable_heat_uses_clm5_inverse_cbrt_not_gsam():
    """DISCRIMINATOR vs the gSAM/LSM4 sibling: production's very-unstable heat
    correction is CLM5's INVERSE cube-root ``zetat^-1/3 - (-zeta)^-1/3`` — it
    matches the CLM5 oracle and DIFFERS materially from a gSAM growing-cbrt
    ``(-zeta)^1/3 - zetat^1/3`` variant.  Evaluated deep in free convection
    (zeta=-100) where the cube-root branches diverge strongly."""
    z0h, zeta = 0.1, -100.0                                # deep free convection
    obu = _ZLDIS/zeta
    got = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obu), _a(z0h)))
    clm5 = _ch_oracle(_ZLDIS, z0h, obu)
    # IDEALIZED growing-cube-root comparator (the on-disk gSAM uses the truncated
    # exponent 0.3333_DBL; immaterial to this >30% structural discriminator).
    d_gsam = (math.log(-_O_ZETAT*obu/z0h) - _psih(-_O_ZETAT) + _psih(z0h/obu)
              + _O_HEAT_CONV*((-zeta)**(1.0/3.0) - _O_ZETAT**(1.0/3.0)))
    ch_gsam = _O_KAPPA/d_gsam
    assert got == pytest.approx(clm5, rel=1e-9, abs=0.0)   # CLM5 form is what runs
    assert abs(got - ch_gsam)/abs(clm5) > 0.3              # growing-cbrt form differs >30%


def test_kb_minus_one_is_zero_not_two_departure():
    """DEPARTURE canary: the module uses kB^-1 = 0 (z0h = z0m).  In the neutral
    limit the heat resistance for the exact kB^-1 = 2 choice (z0h = z0m/e^2, i.e.
    ln(z0m/z0h) = 2) is LARGER than for z0h = z0m by exactly ln(e^2) = 2; and
    monin_obukhov_stability wires z0h = z0m (the smaller-resistance / larger-ch
    branch).  (The on-disk gSAM approximates e^-2 by zt0 = 0.135 z0, a
    ln(1/0.135) = 2.0025 increment.)"""
    z0m = 0.1
    obu = _ZLDIS/1e-6                                       # near neutral
    ch_kb0 = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obu), _a(z0m)))
    ch_kb2 = float(_temperature_humidity_relation(_a(_ZLDIS), _a(obu), _a(z0m/math.e**2)))
    # 1/ch is the neutral resistance ln(z/z0h); the kB^-1=2 choice adds ln(e^2)=2
    assert (_O_KAPPA/ch_kb2 - _O_KAPPA/ch_kb0) == pytest.approx(2.0, rel=1e-4)
    # the solver's rah is consistent with kB^-1 = 0, not kB^-1 = 2
    ur, Ta, Tv, Tc = 4.0, 300.0, 300.5, 301.0              # unstable column
    ustar, rah, raw, uav, zeta = monin_obukhov_stability(
        _a(ur), _a(Ta), _a(Tv), _a(Tc), _a(0.01), _a(0.011), _a(_ZLDIS), _a(z0m),
        n_iters=40, zeta_cap_width=_ZW)
    ch_solver = 1.0/(float(rah)*float(ustar))
    ch_at_zeta_kb0 = _ch_oracle(_ZLDIS, z0m, _ZLDIS/float(zeta))
    ch_at_zeta_kb2 = _ch_oracle(_ZLDIS, z0m/math.e**2, _ZLDIS/float(zeta))
    assert abs(ch_solver - ch_at_zeta_kb0) < abs(ch_solver - ch_at_zeta_kb2)


# --- initialisation + solver ---------------------------------------------------

def test_bulk_richardson_init_matches_zeng1998():
    """_monin_obukhov_init reproduces the Zeng-1998 bulk-Ri first guess."""
    ur, Tv_atm, dthv, z0m = 3.0, 300.0, 1.2, 0.1           # stable (dthv>0)
    um_got, obu_got = _monin_obukhov_init(_a(ur), _a(Tv_atm), _a(dthv), _a(_ZLDIS), _a(z0m), _ZW)
    g = float(constants.g)
    um = max(ur, 0.1)                                       # dthv>=0 branch
    rib = g*_ZLDIS*dthv/(Tv_atm*um**2)
    zeta_raw = rib*math.log(_ZLDIS/z0m)/(1.0 - _O_BETA*min(rib, _O_RIB_MAX))
    # Deliberate departure from CLM5's hard min(zeta, 0.5): a smooth min of
    # width w (a hard kink stalls the canopy Newton solve).  This stable column
    # drives the cap active (raw > 0.5), so it pins both the formula and that
    # the departure from CLM5 stays inside w*ln2.
    w = _ZW
    zeta = zeta_raw - w*math.log1p(math.exp((zeta_raw - _O_ZETA_MAX_STABLE)/w))
    assert zeta_raw > _O_ZETA_MAX_STABLE
    assert _O_ZETA_MAX_STABLE - w*math.log(2.0) < zeta < _O_ZETA_MAX_STABLE
    assert float(um_got) == pytest.approx(um, rel=1e-9, abs=0.0)
    assert float(obu_got) == pytest.approx(_ZLDIS/zeta, rel=1e-9, abs=0.0)


def test_monin_obukhov_init_unstable_gustiness():
    """Unstable init adds convective gustiness um = sqrt(ur^2 + wc^2), wc=0.5."""
    ur, Tv_atm, dthv, z0m = 2.0, 300.0, -0.8, 0.1          # unstable (dthv<0)
    um_got, _ = _monin_obukhov_init(_a(ur), _a(Tv_atm), _a(dthv), _a(_ZLDIS), _a(z0m), _ZW)
    assert float(um_got) == pytest.approx(math.sqrt(ur**2 + 0.5**2), rel=1e-9, abs=0.0)


@pytest.mark.parametrize("Tc", [301.0, 299.0])             # unstable / stable column
def test_solver_converges_full_state(Tc):
    """monin_obukhov_stability reaches a fixed point: the FULL returned state
    (ustar, rah, raw, uav, zeta) at n_iters=8 already matches n_iters=40 to a
    tight tolerance, and the state change across a LATER block of iterations
    (16 vs 8) is smaller than across an EARLIER block (4 vs 2) — contracting, not
    oscillating."""
    z0m = 0.1
    args = (_a(4.0), _a(300.0), _a(300.5), _a(Tc), _a(0.01), _a(0.011), _a(_ZLDIS), _a(z0m))

    def _state(k):
        return jnp.array([float(x) for x in monin_obukhov_stability(*args, n_iters=k, zeta_cap_width=_ZW)])

    s2, s4, s8, s16, s40 = (_state(k) for k in (2, 4, 8, 16, 40))
    for got, exp in zip(s8.tolist(), s40.tolist()):        # all 5 outputs, not just ustar
        assert got == pytest.approx(exp, rel=1e-6)
    # later-block change (16 vs 8) is smaller than earlier-block (4 vs 2): contracting
    assert float(jnp.linalg.norm(s16 - s8)) < float(jnp.linalg.norm(s4 - s2))


# --- auxiliary CLM5 / DifferBESS forms -----------------------------------------

def test_aerodynamics_egvf_blend_matches_oracle():
    """compute_aerodynamics reproduces the LAI-blended z0m / displacement."""
    hc, LAI, rz0m, rd = 12.0, 3.5, 0.055, 0.67
    z0m, displa = compute_aerodynamics(_a(hc), _a(LAI), _a(rz0m), _a(rd))
    lai_f = min(max(LAI, 0.0), 2.0)
    egvf = (1.0 - math.exp(-lai_f)) / (1.0 - math.exp(-2.0))
    exp_displa = hc*rd*egvf
    exp_z0m = math.exp(egvf*math.log(max(hc*rz0m, _Z0MG_BARE)) + (1.0 - egvf)*math.log(_Z0MG_BARE))
    assert float(displa) == pytest.approx(exp_displa, rel=1e-9, abs=0.0)
    assert float(z0m) == pytest.approx(exp_z0m, rel=1e-9, abs=0.0)


def test_below_canopy_cs_blend_matches_oracle():
    """compute_below_canopy_resistance reproduces the clumping-weighted Cs form."""
    uav, CI, LAI = 1.5, 0.8, 3.0
    res, res2 = compute_below_canopy_resistance(_a(uav), _a(CI), _a(LAI))
    csbare = float(constants.kappa_vk)/_CS_BARE_COEF * (
        _Z0MG_BARE*max(uav, 1e-3)/_NU_AIR)**(-_CS_BARE_EXP)
    w = math.exp(-0.5*CI*LAI)
    cs = csbare*w + _CS_DENSE*(1.0 - w)
    exp = 1.0/max(cs*uav, 1e-9)
    assert float(res) == pytest.approx(exp, rel=1e-9, abs=0.0)
    assert float(res2) == float(res)


def test_boundary_layer_rb_matches_oracle():
    """compute_boundary_layer_resistance reproduces the forced-convection rb with
    the sunlit/shaded LAI floors."""
    uav, LAI, fSun, cv, d_leaf = 2.0, 4.0, 0.6, 0.0135, 0.025
    rb_sun, rb_sh = compute_boundary_layer_resistance(
        _a(uav), _a(LAI), _a(fSun), _a(cv), _a(d_leaf))
    rb = 1.0/(cv*math.sqrt(max(uav/d_leaf, 1e-9)))
    assert float(rb_sun) == pytest.approx(rb/max(LAI*fSun, 1e-6), rel=1e-9, abs=0.0)
    assert float(rb_sh) == pytest.approx(rb/max(LAI*(1.0 - fSun), 1e-6), rel=1e-9, abs=0.0)


# --- AD-safety -----------------------------------------------------------------

@pytest.mark.parametrize("name,obu", _REGIMES)
def test_ustar_grad_wrt_um_is_kappa_over_fm(name, obu):
    """ustar = kappa u / fm is linear in u, so dustar/du = ustar/u > 0 in every
    regime (finite, non-vacuous), incl. through the discarded-branch floors."""
    z0m, um = 0.1, 4.0
    u0 = _a(um)
    ustar = float(_friction_velocity(_a(_ZLDIS), _a(z0m), _a(obu), u0))
    g = jax.grad(lambda u: _friction_velocity(_a(_ZLDIS), _a(z0m), _a(obu), u))(u0)
    assert bool(jnp.isfinite(g)) and float(g) > 0.0
    assert float(g) == pytest.approx(ustar/um, rel=1e-9, abs=0.0)


def test_solver_grad_finite_x64_and_float32():
    """grad of ustar wrt the wind speed through the full scan (both a stable and
    an unstable column, exercising the discarded gust-cbrt / where floors) is
    finite and positive in x64 and float32."""
    z0m = 0.1

    def _check():
        for Tc in (301.0, 299.0):                          # unstable / stable
            args = lambda ur: monin_obukhov_stability(
                ur, _a(300.0), _a(300.5), _a(Tc), _a(0.01), _a(0.011),
                _a(_ZLDIS), _a(z0m), n_iters=8, zeta_cap_width=_ZW)[0]
            gr = jax.grad(args)(_a(4.0))
            assert bool(jnp.isfinite(gr)) and float(gr) > 0.0

    _check()
    jax.config.update("jax_enable_x64", False)
    _check()   # autouse fixture restores the entry state afterwards
