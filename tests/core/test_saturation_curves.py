"""Direct tests for the thermo saturation mixing-ratio curves — the shared
``e_sat -> q_sat`` conversion and the Goff variant added for the #762 air-sea
thermodynamic convention (0.98 x Goff surface q_sat)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_mixing_ratio_goff,
    saturation_vapor_pressure,
    saturation_vapor_pressure_goff,
    _mixing_ratio_from_esat,
)

_T = jnp.array([275.0, 285.0, 295.0, 305.0])   # SST range [K]
_P = jnp.full_like(_T, 1.0e5)                    # ~surface [Pa]


def test_tetens_is_shared_conversion_over_tetens_curve():
    """The refactor is exact: saturation_mixing_ratio == the shared
    conversion applied to the Tetens e_sat (no numeric drift)."""
    a = np.asarray(saturation_mixing_ratio(_T, _P))
    b = np.asarray(_mixing_ratio_from_esat(saturation_vapor_pressure(_T), _P))
    np.testing.assert_array_equal(a, b)


def test_goff_is_shared_conversion_over_goff_curve():
    a = np.asarray(saturation_mixing_ratio_goff(_T, _P))
    b = np.asarray(
        _mixing_ratio_from_esat(saturation_vapor_pressure_goff(_T), _P))
    np.testing.assert_array_equal(a, b)


def test_goff_differs_from_tetens_but_close():
    """Goff and Tetens are distinct curves (non-vacuous switch) yet agree
    to a few percent across the SST range — the ~0.5-1 % Δq class the
    #762 convention targets, not a blunder."""
    tet = np.asarray(saturation_mixing_ratio(_T, _P))
    gof = np.asarray(saturation_mixing_ratio_goff(_T, _P))
    assert not np.allclose(tet, gof)                     # actually differ
    rel = np.abs(gof - tet) / tet
    assert np.all(rel < 0.05)                            # same ballpark
    assert np.all(np.isfinite(gof)) and np.all(gof > 0)  # physical


def test_goff_reference_values():
    """Anchor the Goff q_sat to an INDEPENDENT reference (codex round-20:
    'different but nearby' can't catch a wrong liquid curve).  Goff (1957)
    e_sat over liquid at 1 atm: 872.3 Pa at 5 degC (278.15 K), 2338.3 Pa
    at 20 degC (293.15 K); q_sat = eps*e/(p-e) at p=101325 Pa.  Compared
    to the SMOOTH-CAPPED model conversion, so a generous 3% tol absorbs
    the softplus floor/cap (negligible far from p)."""
    from legoesm import constants
    # Published Goff(1957) liquid e_sat [Pa] at standard-atm p [Pa] — the
    # independent reference (not model constants).
    p = 101325.0
    for T_K, e_ref in ((278.15, 872.3), (293.15, 2338.3)):
        q_ref = constants.epsilon * e_ref / (p - e_ref)
        q = float(saturation_mixing_ratio_goff(
            jnp.array([T_K]), jnp.array([p]))[0])
        assert abs(q - q_ref) / q_ref < 0.03, (
            f"Goff q_sat at {T_K} K: {q:.5e} vs ref {q_ref:.5e}")


def test_goff_monotone_in_T():
    gof = np.asarray(saturation_mixing_ratio_goff(_T, _P))
    assert np.all(np.diff(gof) > 0)                      # warmer -> wetter


def test_goff_differentiable():
    """Reverse-mode AD flows through the Goff q_sat (surface flux path)."""
    g = jax.grad(lambda t: saturation_mixing_ratio_goff(
        jnp.array([t]), jnp.array([1.0e5]))[0])(295.0)
    assert np.isfinite(float(g)) and float(g) > 0.0


def test_cam_goff_gratch_saturated_branch_and_gradients():
    """CAM ``wv_sat_svp_to_qsat`` (wv_sat_methods.F90:180-193) at the default
    (smooth, RULE 5) cap, away from ``e_sat = p``: q_sat = 1 where e_sat >> p
    and the epsilo/omeps form where e_sat << p, both to rounding.  The values
    themselves are checked against the RUNNING Fortran in the MG2 oracle
    fixtures; here reverse-mode AD is finite in fp64 and fp32."""
    from legoesm import constants
    from legoesm.thermo import (
        saturation_specific_humidity_goff_gratch,
        saturation_specific_humidity_ice_goff_gratch,
        saturation_vapor_pressure_goff_gratch,
    )
    T = jnp.array([300.0, 300.0])
    p = jnp.array([1000.0, 1.0e5])                 # e_sat(300 K) ~ 3.5e3 Pa
    q = np.asarray(saturation_specific_humidity_goff_gratch(T, p))
    assert q[0] == 1.0
    es = float(saturation_vapor_pressure_goff_gratch(T)[1])
    eps = constants.epsilon
    assert q[1] == pytest.approx(eps * es / (1.0e5 - (1.0 - eps) * es), rel=1e-15)
    for fn in (saturation_specific_humidity_goff_gratch,
               saturation_specific_humidity_ice_goff_gratch):
        for dt in (jnp.float64, jnp.float32):
            g = jax.grad(lambda t: jnp.sum(fn(t, jnp.full(3, 8.0e4, dt))))(
                jnp.array([200.0, 250.0, 270.0], dt))
            assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all(g > 0.0))


# --- RULE 5: CAM q_sat = 1 cap (limiter class), smooth by default ---
_CAP_CASES = [(600.0, 1.0e5), (50.0, 60.0), (99.5, 100.0), (100.0, 100.0), (150.0, 100.0)]


def _cam_svp_to_qsat(es, p):
    from legoesm import constants
    eps = constants.epsilon
    return 1.0 if p - es <= 0.0 else eps * es / (p - (1.0 - eps) * es)


@pytest.mark.parametrize("es,p", _CAP_CASES)
def test_cam_qsat_cap_hard_limit_is_cam(es, p):
    import math
    jax.config.update("jax_enable_x64", True)

    from legoesm.thermo import _specific_humidity_from_esat_cam
    got = float(_specific_humidity_from_esat_cam(jnp.float64(es), jnp.float64(p),
                                                 cap_sharpness=math.inf))
    assert got == _cam_svp_to_qsat(es, p)  # bitwise: same IEEE ops as CAM


# Default-sharpness departure from CAM, per case (KNOWN DIFFERENCE).
_CAP_DEFAULT_GAP = {
    (600.0, 1.0e5): 0.0, (50.0, 60.0): 0.0, (99.5, 100.0): -1.0731362096172425e-05,
    (100.0, 100.0): -1.1139510508489936e-03, (150.0, 100.0): 0.0,
}


@pytest.mark.parametrize("es,p", _CAP_CASES)
def test_cam_qsat_cap_default_gap_is_pinned(es, p):
    from legoesm.thermo import _specific_humidity_from_esat_cam
    jax.config.update("jax_enable_x64", True)
    gap = (float(_specific_humidity_from_esat_cam(jnp.float64(es), jnp.float64(p)))
           - _cam_svp_to_qsat(es, p))
    assert gap == pytest.approx(_CAP_DEFAULT_GAP[(es, p)], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("es,p", _CAP_CASES)
def test_cam_qsat_cap_default_gradients(es, p):
    from jax.test_util import check_grads

    from legoesm.thermo import _specific_humidity_from_esat_cam
    jax.config.update("jax_enable_x64", True)
    check_grads(_specific_humidity_from_esat_cam, (jnp.float64(es), jnp.float64(p)),
                order=2, modes=("rev",))


# --- RULE 5: CAM qsat-table [tmin, tmax] clamp (limiter class), smooth by default ---
# Relative departure of the default from CAM's hard clamp, per (T [K], p [Pa]).
_CLAMP_DEFAULT_REL_GAP = {
    (100.0, 6.0e4): 7.227809275300596e-13,
    (127.16, 6.0e4): 0.312004834823652,
    (130.0, 6.0e4): 0.018059983190437317,
    (250.0, 6.0e4): 0.0,
    (374.0, 1.0e6): -0.009976601552033509,
    (375.16, 1.0e6): -0.025094240926450913,
    (400.0, 1.0e6): -5.91245026142207e-13,
    (300.0, 3.0e3): 0.0,  # p < e_sat: CAM's q = 1 (the cap, not the clamp)
}


def _cam_table_qsat(T, p):
    """CAM qsat (wv_saturation.F90:458-503) on the 1-K svp_trans table."""
    import math

    from legoesm import constants
    from legoesm.thermo import saturation_vapor_pressure_trans
    tmin, tmax = 127.16, 375.16
    t = max(min(T, tmax) - tmin, 0.0)
    i = math.floor(t)
    w = t - i
    e0 = float(saturation_vapor_pressure_trans(jnp.float64(tmin + i)))
    e1 = float(saturation_vapor_pressure_trans(jnp.float64(tmin + i + 1)))
    es = (1.0 - w) * e0 + w * e1
    eps = constants.epsilon
    return 1.0 if p - es <= 0.0 else eps * es / (p - (1.0 - eps) * es)


@pytest.mark.parametrize("T,p", list(_CLAMP_DEFAULT_REL_GAP))
def test_cam_table_clamp_hard_limit_is_cam(T, p):
    import math

    from legoesm.thermo import saturation_specific_humidity_cam_table
    jax.config.update("jax_enable_x64", True)
    got = float(saturation_specific_humidity_cam_table(jnp.float64(T), jnp.float64(p),
                                                       clamp_sharpness=math.inf,
                                                       cap_sharpness=math.inf))
    assert got == _cam_table_qsat(T, p)  # bitwise


@pytest.mark.parametrize("T,p", list(_CLAMP_DEFAULT_REL_GAP))
def test_cam_table_clamp_default_gap_is_pinned(T, p):
    from legoesm.thermo import saturation_specific_humidity_cam_table
    jax.config.update("jax_enable_x64", True)
    ref = _cam_table_qsat(T, p)
    rel = float(saturation_specific_humidity_cam_table(jnp.float64(T), jnp.float64(p))) / ref - 1.0
    assert rel == pytest.approx(_CLAMP_DEFAULT_REL_GAP[(T, p)], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("T", [126.0, 127.16 + 5.0e-5, 200.3, 375.16 - 5.0e-5, 376.0])
def test_cam_table_clamp_default_gradients(T):
    from jax.test_util import check_grads

    from legoesm.thermo import saturation_specific_humidity_cam_table
    jax.config.update("jax_enable_x64", True)
    check_grads(lambda t: saturation_specific_humidity_cam_table(t, jnp.float64(1.0e6)),
                (jnp.float64(T),), order=2, modes=("rev",), eps=1e-4)


@pytest.mark.parametrize("fn", ["saturation_vapor_pressure_goff_gratch",
                                "saturation_vapor_pressure_ice_goff_gratch",
                                "saturation_specific_humidity_goff_gratch",
                                "saturation_specific_humidity_ice_goff_gratch"])
def test_goff_gratch_returns_input_precision_computed_in_fp64(fn):
    """CAM evaluates Goff-Gratch in r8: an fp32 caller (x64 on) gets the
    fp64 value cast to fp32, not an fp64 array leaking into its state."""
    import legoesm.thermo as th
    f = getattr(th, fn)
    with jax.enable_x64(True):  # the fp64 reference must really be fp64
        args64 = [jnp.linspace(190.0, 310.0, 13)]
        if "humidity" in fn:
            args64.append(jnp.linspace(2.0e3, 1.0e5, 13))
        args32 = [a.astype(jnp.float32) for a in args64]
        out32 = f(*args32)
        assert out32.dtype == jnp.float32
        ref = f(*[a.astype(jnp.float64) for a in args32]).astype(jnp.float32)
        np.testing.assert_array_equal(np.asarray(out32), np.asarray(ref))
        ints = [300] + ([100000] if "humidity" in fn else [])
        assert jnp.issubdtype(f(*ints).dtype, jnp.floating)
        assert float(f(*ints)) > 0.0


def test_cam_table_is_built_in_fp64_then_cast():
    """Follow-up row 4: CAM builds estbl in r8 (wv_saturation.F90:226); an
    fp32 run reads that table cast once, not a table evaluated in fp32.  The
    first build happens inside a jit trace, under fp32."""
    import numpy as np

    from legoesm import thermo
    thermo._cam_svp_table_fp64.cache_clear()
    with jax.enable_x64(False):
        q32 = jax.jit(lambda t: thermo.saturation_specific_humidity_cam_table(
            t, jnp.float32(8.0e4)))(jnp.float32(250.0))
        t32 = np.asarray(thermo.cam_svp_table(jnp.float32))
        direct32 = np.asarray(thermo.saturation_vapor_pressure_trans(
            127.16 + jnp.arange(t32.size, dtype=jnp.float32)))
    with jax.enable_x64(True):
        t64 = np.asarray(thermo.cam_svp_table(jnp.float64))
        direct64 = np.asarray(thermo.saturation_vapor_pressure_trans(
            127.16 + jnp.arange(t64.size, dtype=jnp.float64)))
    assert q32.dtype == jnp.float32 and np.isfinite(q32)
    assert t32.dtype == np.float32 and t64.dtype == np.float64
    np.testing.assert_array_equal(t64, direct64)
    np.testing.assert_array_equal(t32, t64.astype(np.float32))
    assert np.any(direct32 != t32)  # the fp32-evaluated table differs


def test_cam_qsat_reads_the_fp64_built_table_in_fp32(monkeypatch):
    """Follow-up row 4, consumer side: an fp32 qsat interpolates the
    fp64-built table cast to fp32, not an fp32-evaluated table (the two give
    different q somewhere on a 0.1 K sweep)."""
    import numpy as np

    from legoesm import thermo
    with jax.enable_x64(False):
        T = jnp.arange(180.0, 320.0, 0.1, dtype=jnp.float32)
        p = jnp.float32(8.0e4)
        n = thermo.cam_svp_table(jnp.float32).shape[0]
        direct32 = thermo.saturation_vapor_pressure_trans(
            127.16 + jnp.arange(n, dtype=jnp.float32))
        q = np.asarray(thermo.saturation_specific_humidity_cam_table(T, p))
        cast64 = thermo.cam_svp_table(jnp.float32)
        monkeypatch.setattr(thermo, "cam_svp_table", lambda dtype: cast64)
        q_cast = np.asarray(thermo.saturation_specific_humidity_cam_table(T, p))
        monkeypatch.setattr(thermo, "cam_svp_table", lambda dtype: direct32)
        q_fp32_table = np.asarray(thermo.saturation_specific_humidity_cam_table(T, p))
    np.testing.assert_array_equal(q, q_cast)
    assert np.any(q != q_fp32_table)
