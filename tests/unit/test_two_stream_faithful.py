"""Faithfulness pins for the RRTMGP two-stream RTE solver (general k != 0).

Oracle: the PUBLISHED two-stream equations, transcribed independently in NumPy --
the swirl_jatmos / rte-rrtmgp source this was ported from is NOT on disk (only the
swirl_jatmos copyright header remains). The published forms are cross-checked
against ECMWF ecRad ``radiation_two_stream.F90`` (an independent, equivalent
Meador-Weaver/Zdunkowski implementation), cited per subroutine below. These pins
call PRODUCTION (``monochromatic_two_stream.sw_cell_properties`` /
``lw_cell_source_and_properties``) on a crafted single layer and compare against
an independent NumPy reimpl of the cited closed forms:

* SW two-stream coefficients + diffuse R/T -- Meador & Weaver (1980), J. Atmos.
  Sci. 37, 630-643, Table 1 (the "practical improved" / Zdunkowski PIFM row) and
  Eqs. 25-26; gamma3 cross-checked against rte-rrtmgp ``sw_two_stream`` and ecRad
  ``calc_two_stream_gammas_sw``;
* SW direct-beam R/T -- Meador & Weaver (1980) Eqs. 14-15, built from the CANONICAL
  e^{+k*tau} form (NOT production's e^{-k*tau} refactor, so a shared refactor bug
  cannot hide); cross-verified against ecRad ``calc_reflectance_transmittance_sw``.
  Note the two conventions: production ``r_dir`` = ecRad ``ref_dir/mu0`` (mu0 folded
  downstream), and production ``t_dir`` is the SCATTERED part = M-W Eq.15 total minus
  the separately-transported unscattered beam t0;
* LW two-stream coefficients + diffuse R/T -- Fu et al. (1997) diffusivity form
  (secant D = 1.66) used by rte-rrtmgp ``lw_two_stream`` / ecRad;
* LW linear-in-tau Planck source -- Toon et al. (1989) Eqs. 26-27.

This complements ``tests/unit/test_two_stream_conservative_limit.py``, which pins
ONLY the ssa=1 conservative-scattering k->0 limit (gamma1==gamma2, k=0). Here the
inputs are deliberately ABSORBING (ssa<1, k>0) so the pins constrain gamma1..4,
the eigenvalue k, and the Meador-Weaver denominator independently -- the regime
that was previously untested against any published form.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
    monochromatic_two_stream as mts,
)

_SHAPE = (1, 1, 3)


def _fill(val, dtype=jnp.float64):
    return jnp.full(_SHAPE, val, dtype=dtype)


# ---------------------------------------------------------------------------
# SW: Meador-Weaver (1980) coefficients + diffuse R/T, general k != 0
# ---------------------------------------------------------------------------

def _mw_sw_gammas(ssa, g, mu0):
    """Independent NumPy transcription of the SW two-stream coefficients:
    Meador-Weaver (1980) Table 1 "practical improved" (== Zdunkowski PIFM ==
    rte-rrtmgp sw_two_stream)."""
    gamma1 = 0.25 * (8.0 - ssa * (5.0 + 3.0 * g))
    gamma2 = 0.25 * 3.0 * ssa * (1.0 - g)
    gamma3 = 0.25 * (2.0 - 3.0 * mu0 * g)
    gamma4 = 1.0 - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4
    return gamma1, gamma2, gamma3, gamma4, alpha1, alpha2


def _mw_diffuse_rt(gamma1, gamma2, tau):
    """Meador-Weaver (1980) Eqs. 25-26 diffuse reflectance/transmittance."""
    k = np.sqrt((gamma1 + gamma2) * (gamma1 - gamma2))
    e2 = np.exp(-2.0 * tau * k)
    denom = k * (1.0 + e2) + gamma1 * (1.0 - e2)
    r = gamma2 * (1.0 - e2) / denom
    t = 2.0 * k * np.exp(-tau * k) / denom
    return k, r, t


def test_sw_diffuse_rt_matches_meador_weaver_general_k():
    """Absorbing SW layer (ssa<1 => k>0): production r_diff/t_diff equal the
    independent Meador-Weaver Eqs. 25-26 oracle to f64 tolerance.

    Non-vacuous: this is the GENERAL case (k>0, r+t<1), NOT the ssa=1 k->0 limit
    the sibling test covers -- so it genuinely constrains gamma1, gamma2 and the
    M-W denominator, not the degenerate gamma1==gamma2 collapse."""
    ssa_v, g_v, tau_v, zenith = 0.7, 0.85, 1.0, 0.6
    mu0 = float(np.cos(zenith))
    g1, g2, *_ = _mw_sw_gammas(ssa_v, g_v, mu0)
    k, r_exp, t_exp = _mw_diffuse_rt(g1, g2, tau_v)

    # Guard the regime: this must be the general k>0, absorbing case.
    assert k > 1e-2, "test degenerated to the k->0 limit; pick a smaller ssa"
    assert 0.0 < r_exp and 0.0 < t_exp
    assert r_exp + t_exp < 0.999, "layer not absorbing; r+t should be < 1"

    props = mts.sw_cell_properties(zenith, _fill(tau_v), _fill(ssa_v), _fill(g_v))
    np.testing.assert_allclose(np.asarray(props["r_diff"]), r_exp, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(props["t_diff"]), t_exp, rtol=1e-12)


# ---------------------------------------------------------------------------
# SW: Meador-Weaver (1980) direct-beam R/T, Eqs. 14-15 (constrains gamma3/4)
# ---------------------------------------------------------------------------

def _mw_direct_rt_canonical(ssa, g, tau, mu0):
    """Independent CANONICAL Meador-Weaver (1980) Eqs. 14-15 direct-beam R/T.

    Uses the e^{+k*tau} canonical denominator ``E = (k+g1)e^{k*tau} +
    (k-g1)e^{-k*tau}`` -- STRUCTURALLY DISTINCT from production's
    e^{-k*tau}-multiplied refactor (monochromatic_two_stream.py:229-268), so a
    shared refactor bug cannot hide.  Cross-verified against ECMWF ecRad
    ``radiation_two_stream.F90::calc_reflectance_transmittance_sw`` (``ref_dir``
    L517-520, ``trans_dir_diff`` L525-528).

    Convention: production (like ecRad) defines the incoming direct flux
    perpendicular to the beam; ecRad bakes a ``mu0`` factor into the coefficient
    (F90:513) while production folds ``mu0`` into ``flux_down_direct`` downstream
    (``sw_cell_source``).  So production's ``r_dir`` equals ecRad ``ref_dir/mu0``
    -- i.e. the prefactor here carries NO ``mu0``.  Meador-Weaver Eq. 15 is the
    TOTAL direct transmittance; production ``t_dir`` is only its SCATTERED part
    (the unscattered ``t0`` is transported separately), so the oracle returns
    ``T_total - t0``.
    """
    g1, g2, g3, g4, a1, a2 = _mw_sw_gammas(ssa, g, mu0)
    k = np.sqrt((g1 + g2) * (g1 - g2))
    q = k * mu0
    t0 = np.exp(-tau / mu0)
    ekt = np.exp(k * tau)
    emkt = np.exp(-k * tau)
    e_denom = (k + g1) * ekt + (k - g1) * emkt          # canonical M-W denom
    pref = ssa / ((1.0 - q * q) * e_denom)               # production convention: no mu0
    r_dir = pref * (
        (1.0 - q) * (a2 + k * g3) * ekt
        - (1.0 + q) * (a2 - k * g3) * emkt
        - 2.0 * (k * g3 - a2 * q) * t0
    )
    t_total = t0 - pref * (
        (1.0 + q) * (a1 + k * g4) * ekt * t0
        - (1.0 - q) * (a1 - k * g4) * emkt * t0
        - 2.0 * (k * g4 + a1 * q)
    )
    return t0, r_dir, t_total - t0  # t_dir is the SCATTERED contribution only


def test_sw_direct_beam_rt_matches_meador_weaver_eq14_15():
    """Production r_dir/t_dir equal the independent CANONICAL Meador-Weaver
    Eqs. 14-15 oracle (built from the e^{+k*tau} form, not production's refactor).
    This is the only exact-value direct-beam oracle in this file and the pin that
    independently constrains gamma3, gamma4, alpha1, alpha2 (absent from the
    diffuse R/T): a wrong production gamma3 would fail here.

    The physical clip ``r_dir=clip(.,0,1-t0)``, ``t_dir=clip(.,0,1-t0-r_dir)``
    must be INACTIVE for the chosen input so the pin tests the FORMULA, not the
    clamp -- asserted explicitly below."""
    ssa_v, g_v, tau_v, zenith = 0.7, 0.85, 1.0, 0.6
    mu0 = float(np.cos(zenith))
    t0, r_can, t_can = _mw_direct_rt_canonical(ssa_v, g_v, tau_v, mu0)

    # Clip must be inactive: the raw M-W values sit strictly inside the bounds.
    assert 0.0 < r_can < 1.0 - t0, f"r_dir clip active (r_can={r_can:.4f})"
    assert 0.0 < t_can < 1.0 - t0 - r_can, f"t_dir clip active (t_can={t_can:.4f})"

    props = mts.sw_cell_properties(zenith, _fill(tau_v), _fill(ssa_v), _fill(g_v))
    np.testing.assert_allclose(np.asarray(props["r_dir"]), r_can, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(props["t_dir"]), t_can, rtol=1e-12)


# ---------------------------------------------------------------------------
# LW: Fu (1997) diffusivity coefficients + diffuse R/T, general k != 0
# ---------------------------------------------------------------------------

def test_lw_diffuse_rt_matches_fu1997_diffusivity_general_k():
    """Absorbing LW layer: production r_diff/t_diff (Fu 1997 diffusivity form,
    secant D=1.66) equal the independent oracle. The LW gamma1/gamma2 use the
    diffusivity closure, NOT the SW Zdunkowski one -- this pins that closure in
    the general k>0 regime (the sibling test covers only the ssa=1 k->0 limit)."""
    ssa_v, g_v, tau_v = 0.4, 0.5, 2.0
    d = float(mts._LW_DIFFUSIVE_FACTOR)
    assert d == 1.66, "Fu-Liou diffusivity secant drifted from 1.66"
    g1 = d * (1.0 - 0.5 * ssa_v * (1.0 + g_v))
    g2 = d * 0.5 * ssa_v * (1.0 - g_v)
    k, r_exp, t_exp = _mw_diffuse_rt(g1, g2, tau_v)
    assert k > 1e-2 and (r_exp + t_exp) < 0.999  # general absorbing regime

    out = mts.lw_cell_source_and_properties(
        _fill(tau_v), _fill(ssa_v), _fill(1.0), _fill(1.0), _fill(g_v),
    )
    np.testing.assert_allclose(np.asarray(out["r_diff"]), r_exp, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out["t_diff"]), t_exp, rtol=1e-12)


# ---------------------------------------------------------------------------
# LW: Toon (1989) linear-in-tau Planck source, Eqs. 26-27
# ---------------------------------------------------------------------------

def test_lw_linear_in_tau_planck_source_matches_toon1989():
    """Production src_up/src_down equal the independent Toon (1989) Eqs. 26-27
    linear-in-tau Planck source. The Planck Taylor slope over the layer is
    (B_bot - B_top)/tau; b1 is that slope divided by (g1+g2) (the transformed
    source coefficient): b1 = (B_bot - B_top)/(tau*(g1+g2)). Face sources
    c_up = B +/- b1, cell-centre residual src = pi*(out - r*in_down - t*in_up).

    Uses DISTINCT top/bottom Planck values (b1 != 0, a non-trivial gradient) and
    tau well above the LW-source mask floor so the source is not zeroed."""
    ssa_v, g_v, tau_v = 0.4, 0.5, 2.0
    src_top, src_bot = 3.0, 5.0  # W/m^2/sr, distinct => b1 != 0
    d = float(mts._LW_DIFFUSIVE_FACTOR)
    g1 = d * (1.0 - 0.5 * ssa_v * (1.0 + g_v))
    g2 = d * 0.5 * ssa_v * (1.0 - g_v)
    _, r, t = _mw_diffuse_rt(g1, g2, tau_v)

    b1 = (src_bot - src_top) / (tau_v * (g1 + g2))
    assert abs(b1) > 1e-3, "b1 vanished; choose distinct top/bottom Planck values"
    c_up_top, c_up_bot = src_top + b1, src_bot + b1
    c_dn_top, c_dn_bot = src_top - b1, src_bot - b1
    # cell_center_src_fn(downstream_out, downstream_in, upstream_in, r, t, tau)
    src_up_exp = np.pi * (c_up_top - r * c_dn_top - t * c_up_bot)
    src_dn_exp = np.pi * (c_dn_bot - r * c_up_bot - t * c_dn_top)

    out = mts.lw_cell_source_and_properties(
        _fill(tau_v), _fill(ssa_v), _fill(src_bot), _fill(src_top), _fill(g_v),
    )
    np.testing.assert_allclose(np.asarray(out["src_up"]), src_up_exp, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(out["src_down"]), src_dn_exp, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-check: gamma3 uses the rte-rrtmgp (2 - 3*mu0*g)/4 form (mu0-dependent)
# ---------------------------------------------------------------------------

def test_sw_gamma3_is_mu0_dependent_rte_rrtmgp_form():
    """gamma3 = (2 - 3*mu0*g)/4 (rte-rrtmgp sw_two_stream) enters ONLY the direct
    beam, so r_dir must change with the zenith angle at fixed (tau, ssa, g) --
    a discriminator that the mu0-dependent gamma3 (not a mu0-independent constant)
    is actually wired into the direct-beam solution."""
    ssa_v, g_v, tau_v = 0.7, 0.85, 1.0
    r_lo = mts.sw_cell_properties(0.2, _fill(tau_v), _fill(ssa_v), _fill(g_v))["r_dir"]
    r_hi = mts.sw_cell_properties(1.2, _fill(tau_v), _fill(ssa_v), _fill(g_v))["r_dir"]
    assert abs(float(r_lo[0, 0, 0]) - float(r_hi[0, 0, 0])) > 1e-6, (
        "r_dir did not respond to the zenith angle -> gamma3 mu0-dependence "
        "not wired into the direct beam"
    )
