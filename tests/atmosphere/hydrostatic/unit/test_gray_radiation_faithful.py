"""Oracle-faithfulness tests for two-stream gray radiation (``gray.py``).

Oracle: Isca ``two_stream_gray_rad.F90`` (ExeClim/Isca), the ``B_FRIERSON``
branch that implements Frierson, Held & Zurita-Gotor (2006). We port the
oracle's longwave/shortwave/heating loops into a self-contained NumPy
reference (``_isca_frierson_port``) and assert our JAX scheme reproduces it to
round-off *on the matched subset*: ``lw_diff_factor=1``, Isca tau constants,
dry LW (``q_v=None``), black surface (``sfc_emissivity=1``), ``sw_exponent=4``,
Isca ``sw_diff=0`` (our SW has no latitude-diffusivity knob), ``diabatic_acce=1``
(we have no heating-acceleration factor), ``p_s=p_ref`` (so ``p/p_s`` == Isca's
``p/p_std``), and ordinary layer thickness (production floors ``dp`` at 1 Pa in
the heating divide — irrelevant here). It is NOT a general Isca replica; the
subset is what these tests certify.

The scheme ships *re-tuned constants* on top of the faithful forms, so a second
group of tests pins each DEPARTURE (``lw_diff_factor=1.66``, larger tau,
``sw_exponent=2``, ``sw_tau_0=0.22``, the Byrne&O'Gorman-style moisture term)
that the module docstring documents — these must fail loudly if a "fix" quietly
changes them without updating the Faithfulness section.

Convention note: the port uses ``sigma = p/p_s`` (the FHZ06 *paper* form, which
our scheme also uses), which equals Isca code's ``p/p_std`` fixed reference ONLY
because we build the column with ``p_s = p_ref``. This isolates the two-stream
*recurrence + optical-depth forms* from the (separately documented) p_std-vs-p_s
coordinate departure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import (
    _compute_lw_optical_depth,
    gray_radiation,
)

# Isca ``two_stream_gray_rad.F90`` B_FRIERSON namelist defaults (the oracle).
ISCA_IR_TAU_EQ = 6.0
ISCA_IR_TAU_POLE = 1.5
ISCA_LINEAR_TAU = 0.1
ISCA_WV_EXPONENT = 4.0
ISCA_SOLAR_EXPONENT = 4.0
ISCA_ATM_ABS_DEFAULT = 0.0  # Isca's DEFAULT SW atmosphere is transparent.


@pytest.fixture(autouse=True)
def _enable_x64():
    """Run these numeric faithfulness checks in float64 (restored after)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# NumPy port of the Isca B_FRIERSON loops (the oracle reference)
# ---------------------------------------------------------------------------
def _isca_frierson_port(
    T,
    p_half,
    T_sfc,
    lat,
    insolation,
    *,
    tau_eq,
    tau_pole,
    linear_tau,
    wv_exp,
    atm_abs,
    solar_exp,
    albedo,
    sw_diff=0.0,
):
    """Direct NumPy port of Isca ``two_stream_gray_rad`` B_FRIERSON.

    Mirrors the B_FRIERSON loops (down: ``F(k+1)=F(k)*t+B(1-t)``; up:
    ``F(k)=F(k+1)*t+B(1-t)`` from a black surface; SW Beer-Lambert down + a
    constant ``albedo*sw_down(sfc)`` up; heating ``g/cp*dF_net_up/dp``). Uses the
    ``sigma=p/p_s`` coordinate, which equals Isca's ``p/p_std`` only when
    ``p_s=p_std`` (as ``_columns`` builds it); ``sw_diff`` defaults to 0 (Isca's
    latitude SW-diffusivity factor, which the scheme cannot represent). All
    arithmetic in float64. Interfaces top(0)->surface(nlev); ``B=sigma*T^4``.
    """
    T = np.asarray(T, dtype=np.float64)
    p_half = np.asarray(p_half, dtype=np.float64)
    T_sfc = np.asarray(T_sfc, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    insolation = np.asarray(insolation, dtype=np.float64)
    ncol, nlev = T.shape
    sig = constants.sigma_sb
    p_s = p_half[:, -1]
    sigma = p_half / p_s[:, None]  # (ncol, nlev+1)

    # --- Longwave optical depth (prescribed dry function of lat & sigma) ---
    tau0 = tau_eq + (tau_pole - tau_eq) * np.sin(lat) ** 2  # (ncol,)
    lw_tau = tau0[:, None] * (
        linear_tau * sigma + (1.0 - linear_tau) * sigma ** wv_exp
    )  # (ncol, nlev+1)
    dtau = lw_tau[:, 1:] - lw_tau[:, :-1]  # (ncol, nlev)
    dtrans = np.exp(-dtau)  # NO diffusivity factor (D=1) in Isca
    B = sig * T ** 4  # (ncol, nlev)

    # --- Downward sweep (top -> bottom) ---
    lw_down = np.zeros((ncol, nlev + 1), dtype=np.float64)
    for k in range(nlev):
        lw_down[:, k + 1] = lw_down[:, k] * dtrans[:, k] + B[:, k] * (1.0 - dtrans[:, k])

    # --- Upward sweep (bottom -> top), black surface ---
    lw_up = np.zeros((ncol, nlev + 1), dtype=np.float64)
    lw_up[:, nlev] = sig * T_sfc ** 4
    for k in range(nlev - 1, -1, -1):
        lw_up[:, k] = lw_up[:, k + 1] * dtrans[:, k] + B[:, k] * (1.0 - dtrans[:, k])

    # --- Shortwave: Beer-Lambert down, constant albedo-reflected up ---
    sw_tau_0 = (1.0 - sw_diff * np.sin(lat) ** 2) * atm_abs  # (ncol,)
    sw_tau = sw_tau_0[:, None] * sigma ** solar_exp
    sw_down = insolation[:, None] * np.exp(-sw_tau)
    sw_up = np.broadcast_to((albedo * sw_down[:, -1])[:, None], sw_down.shape).copy()

    # --- Heating rate: g/cp * d(F_up - F_down)/dp ---
    rad_up = (lw_up - lw_down) + (sw_up - sw_down)
    dF = rad_up[:, 1:] - rad_up[:, :-1]
    dp = p_half[:, 1:] - p_half[:, :-1]
    hr = (constants.g / constants.c_pd) * dF / dp

    return lw_up, lw_down, sw_up, sw_down, hr


# ---------------------------------------------------------------------------
# Test column
# ---------------------------------------------------------------------------
def _columns(nlev=20):
    """Two columns (equator + high latitude), p_s = p_ref, warm surface.

    Returns (T, p_full, p_half, T_sfc, lat, q_v, insolation) as arrays with a
    leading column axis. p_half[:, 0] = 0 (model top) so sum(dtau)=tau0(lat).
    """
    p_s = constants.p_ref  # = p_std, so sigma=p/p_s matches the Isca reference
    sigma_half = np.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half_1d = sigma_half * p_s
    p_full_1d = sigma_full * p_s

    # A stable-ish decreasing-with-height profile.
    T_1d = 300.0 * np.clip(sigma_full, 0.02, None) ** 0.19
    T_1d = np.maximum(T_1d, 200.0)

    T = np.stack([T_1d, T_1d])  # (2, nlev)
    p_full = np.stack([p_full_1d, p_full_1d])
    p_half = np.stack([p_half_1d, p_half_1d])
    T_sfc = np.array([300.0, 295.0])
    lat = np.array([0.0, 1.1])  # equator, ~63 deg
    insolation = np.array([420.0, 180.0])

    # Synthetic positive moisture profile (surface-heavy, decreasing with
    # height) — only the moisture-departure test uses it, and it needs no
    # saturation physics, just q_v > 0.
    q_v = 0.015 * sigma_full[None, :] ** 3 * np.ones((2, 1))

    return T, p_full, p_half, T_sfc, lat, q_v, insolation


def _isca_matched_config():
    """GrayRadiationConfig that reproduces Isca B_FRIERSON for a chosen atm_abs.

    D=1 (no diffusivity), Isca tau constants, black surface, dry LW, Isca
    ``solar_exponent=4``. SW ``sw_tau_0`` maps to Isca ``atm_abs``; we set it to a
    NONZERO 0.3 (not the oracle's default ``atm_abs=0.0``) so the Beer-Lambert SW
    path is actually exercised. The port is fed the SAME ``atm_abs=sw_tau_0``, so
    the full-recurrence test matches the oracle FORM at this atm_abs, not the
    transparent-SW default.
    """
    return GrayRadiationConfig(
        tau_equator=ISCA_IR_TAU_EQ,
        tau_pole=ISCA_IR_TAU_POLE,
        linear_frac=ISCA_LINEAR_TAU,
        tau_moist_coeff=0.0,
        lw_diff_factor=1.0,
        sfc_emissivity=1.0,
        sw_tau_0=0.3,  # = atm_abs; nonzero to exercise SW absorption
        sw_exponent=ISCA_SOLAR_EXPONENT,
        sfc_albedo=0.3,
    )


# ===========================================================================
# FAITHFUL: forms reproduce the Isca B_FRIERSON port to round-off
# ===========================================================================
def test_faithful_full_recurrence_matches_isca_port():
    """LW+SW fluxes and heating reproduce the Isca B_FRIERSON port (D=1)."""
    T, p_full, p_half, T_sfc, lat, _q, insol = _columns()
    cfg = _isca_matched_config()

    out = gray_radiation(
        jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(p_half),
        jnp.asarray(T_sfc), jnp.asarray(lat), None, jnp.asarray(insol), cfg,
    )
    lw_up, lw_down, sw_up, sw_down, hr = _isca_frierson_port(
        T, p_half, T_sfc, lat, insol,
        tau_eq=ISCA_IR_TAU_EQ, tau_pole=ISCA_IR_TAU_POLE,
        linear_tau=ISCA_LINEAR_TAU, wv_exp=ISCA_WV_EXPONENT,
        atm_abs=cfg.sw_tau_0, solar_exp=ISCA_SOLAR_EXPONENT, albedo=cfg.sfc_albedo,
    )

    # LW and SW checked SEPARATELY so a sign error that cancels in the total
    # cannot pass.
    assert np.allclose(np.asarray(out.lw_flux_up), lw_up, rtol=1e-10, atol=1e-9)
    assert np.allclose(np.asarray(out.lw_flux_down), lw_down, rtol=1e-10, atol=1e-9)
    assert np.allclose(np.asarray(out.sw_flux_up), sw_up, rtol=1e-10, atol=1e-9)
    assert np.allclose(np.asarray(out.sw_flux_down), sw_down, rtol=1e-10, atol=1e-9)
    assert np.allclose(np.asarray(out.heating_rate), hr, rtol=1e-9, atol=1e-14)


def test_faithful_lw_latitude_profile_column_optical_depth():
    """sum_k dtau_k == tau0(lat) = tau_eq+(tau_pole-tau_eq)sin^2(lat).

    With p_half[0]=0 the surface cumulative tau equals tau0 exactly (the sigma
    blend is 1 at sigma=1), so the column-integrated LW optical depth isolates
    the Isca ``lw_tau_0`` latitude form.
    """
    T, _pf, p_half, _ts, lat, _q, _insol = _columns()
    cfg = _isca_matched_config()
    p_s = p_half[:, -1]
    dtau = _compute_lw_optical_depth(
        jnp.asarray(p_half), jnp.asarray(p_s), jnp.asarray(lat), None, cfg,
    )
    col_tau = np.asarray(dtau).sum(axis=1)  # (ncol,)
    expected = ISCA_IR_TAU_EQ + (ISCA_IR_TAU_POLE - ISCA_IR_TAU_EQ) * np.sin(lat) ** 2
    assert np.allclose(col_tau, expected, rtol=1e-12, atol=1e-12)
    # equator strictly more opaque than high latitude (tau_eq > tau_pole)
    assert col_tau[0] > col_tau[1]


def test_faithful_lw_vertical_blend_linear_plus_quartic():
    """Cumulative LW tau follows tau0*[f_l*sigma+(1-f_l)*sigma^4] (Isca form)."""
    T, _pf, p_half, _ts, lat, _q, _insol = _columns()
    cfg = _isca_matched_config()
    p_s = p_half[:, -1]
    dtau = np.asarray(
        _compute_lw_optical_depth(
            jnp.asarray(p_half), jnp.asarray(p_s), jnp.asarray(lat), None, cfg,
        )
    )
    # Reconstruct cumulative tau at interfaces and compare to the closed form.
    cum = np.concatenate([np.zeros((dtau.shape[0], 1)), np.cumsum(dtau, axis=1)], axis=1)
    sigma = p_half / p_s[:, None]
    tau0 = ISCA_IR_TAU_EQ + (ISCA_IR_TAU_POLE - ISCA_IR_TAU_EQ) * np.sin(lat) ** 2
    expected = tau0[:, None] * (
        ISCA_LINEAR_TAU * sigma + (1.0 - ISCA_LINEAR_TAU) * sigma ** ISCA_WV_EXPONENT
    )
    assert np.allclose(cum, expected, rtol=1e-12, atol=1e-12)


def test_faithful_sw_upward_constant_albedo_times_surface_down():
    """SW upward flux is height-constant == sfc_albedo*sw_down(sfc) (Isca)."""
    T, p_full, p_half, T_sfc, lat, _q, insol = _columns()
    cfg = _isca_matched_config()
    out = gray_radiation(
        jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(p_half),
        jnp.asarray(T_sfc), jnp.asarray(lat), None, jnp.asarray(insol), cfg,
    )
    sw_up = np.asarray(out.sw_flux_up)
    sw_down = np.asarray(out.sw_flux_down)
    # constant across levels
    assert np.allclose(sw_up, sw_up[:, :1], rtol=0, atol=1e-10)
    # equals albedo * surface-incident downward SW
    assert np.allclose(sw_up[:, 0], cfg.sfc_albedo * sw_down[:, -1], rtol=1e-12, atol=1e-12)


def test_faithful_black_surface_bc_at_unit_emissivity():
    """At sfc_emissivity=1 the surface LW source == sigma*T_sfc^4 (Isca b_surf)."""
    T, p_full, p_half, T_sfc, lat, _q, insol = _columns()
    cfg = _isca_matched_config()  # sfc_emissivity=1.0
    out = gray_radiation(
        jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(p_half),
        jnp.asarray(T_sfc), jnp.asarray(lat), None, jnp.asarray(insol), cfg,
    )
    b_surf = constants.sigma_sb * np.asarray(T_sfc) ** 4
    assert np.allclose(np.asarray(out.lw_flux_up)[:, -1], b_surf, rtol=1e-12, atol=1e-9)


# ===========================================================================
# DEPARTURES: re-tuned constants that differ from Isca B_FRIERSON
# ===========================================================================
def test_departure_default_constants_are_not_isca():
    """Ship defaults are the re-tuned values, NOT the Isca B_FRIERSON namelist.

    Canary: if someone changes these to the Isca values (or anything else) the
    module docstring's Faithfulness section must be updated in the same change.
    """
    d = GrayRadiationConfig()
    assert (d.tau_equator, d.tau_pole, d.linear_frac) == (7.2, 1.8, 0.2)
    assert (d.tau_equator, d.tau_pole, d.linear_frac) != (
        ISCA_IR_TAU_EQ, ISCA_IR_TAU_POLE, ISCA_LINEAR_TAU,
    )
    assert d.lw_diff_factor == 1.66  # Isca applies no separate diffusivity (=1)
    assert d.sw_exponent == 2.0 and d.sw_exponent != ISCA_SOLAR_EXPONENT  # Isca 4
    assert d.sw_tau_0 == 0.22 and d.sw_tau_0 != ISCA_ATM_ABS_DEFAULT  # Isca 0


def test_departure_diffusivity_factor_applied_as_exp_minus_D_dtau():
    """PRODUCTION dark-column LW profile pins the PER-LAYER factor exp(-D*dtau_k).

    Set the atmospheric temperature to 0 (B = sigma*T^4 = 0) with a black
    surface, so the only LW source is the surface and its emission is
    transmitted layer-by-layer: lw_up(interface k) = sigma*T_sfc^4 *
    prod_{j>=k} exp(-D*dtau_j) = sigma*T_sfc^4 * exp(-D*sum_{j>=k} dtau_j).
    Asserting the WHOLE upward-flux profile (every interface, not just TOA)
    against this independent expression certifies the scheme applies exactly
    exp(-D*dtau_k) per layer (Isca uses D=1); no algebra outside the scheme.
    """
    T, p_full, p_half, T_sfc, lat, _q, insol = _columns()
    p_s = p_half[:, -1]
    T_dark = np.zeros_like(T)  # zero atmospheric emission
    dtau = np.asarray(
        _compute_lw_optical_depth(
            jnp.asarray(p_half), jnp.asarray(p_s), jnp.asarray(lat), None,
            _isca_matched_config(),
        )
    )  # (ncol, nlev), D-independent
    # Suffix optical depth from each interface down to the surface (interface
    # nlev has 0). rev[:, k] = sum(dtau[:, k:]).
    rev = np.cumsum(dtau[:, ::-1], axis=1)[:, ::-1]
    tau_below = np.concatenate([rev, np.zeros((dtau.shape[0], 1))], axis=1)  # (ncol, nlev+1)
    b_surf = constants.sigma_sb * T_sfc ** 4

    def _lw_up_profile(D):
        cfg = _isca_matched_config()._replace(lw_diff_factor=D)
        out = gray_radiation(
            jnp.asarray(T_dark), jnp.asarray(p_full), jnp.asarray(p_half),
            jnp.asarray(T_sfc), jnp.asarray(lat), None, jnp.asarray(insol), cfg,
        )
        return np.asarray(out.lw_flux_up)  # (ncol, nlev+1)

    for D in (1.0, 1.66):
        expected = b_surf[:, None] * np.exp(-D * tau_below)
        assert np.allclose(_lw_up_profile(D), expected, rtol=1e-10, atol=1e-9)

    # Corollary: larger D transmits less surface emission at TOA (this dark
    # column has no atmospheric re-emission, so the direction is unambiguous).
    assert np.all(_lw_up_profile(1.66)[:, 0] < _lw_up_profile(1.0)[:, 0])


def test_departure_sw_exponent_used_as_sigma_power():
    """The configured sw_exponent is USED as sw_down=insolation*exp(-tau0*sigma^n).

    Pins production sw_down against an independently computed Beer-Lambert
    profile for n=2 (our default, != Isca 4) and n=4 (Isca), so this verifies
    the exponent is applied as sigma**exponent — not merely that the two differ.
    """
    T, p_full, p_half, T_sfc, lat, _q, insol = _columns()
    base = _isca_matched_config()._replace(sw_tau_0=0.5)  # strong absorption
    sigma = p_half / p_half[:, -1][:, None]

    for n in (2.0, 4.0):
        out = gray_radiation(
            jnp.asarray(T), jnp.asarray(p_full), jnp.asarray(p_half),
            jnp.asarray(T_sfc), jnp.asarray(lat), None, jnp.asarray(insol),
            base._replace(sw_exponent=n),
        )
        expected = insol[:, None] * np.exp(-base.sw_tau_0 * sigma ** n)
        assert np.allclose(np.asarray(out.sw_flux_down), expected, rtol=1e-10, atol=1e-9)

    # Corollary: n=2 and n=4 genuinely differ in the interior (not a no-op knob).
    d2 = insol[:, None] * np.exp(-base.sw_tau_0 * sigma ** 2)
    d4 = insol[:, None] * np.exp(-base.sw_tau_0 * sigma ** 4)
    assert not np.allclose(d2[:, 1:-1], d4[:, 1:-1], rtol=1e-3, atol=1e-2)


def test_departure_moisture_adds_lw_opacity_absent_in_frierson():
    """The Byrne&O'Gorman-style moisture term adds opacity Frierson lacks.

    dtau(moist) = dtau(dry) + tau_moist_coeff*q_v*dp/g, elementwise > dry where
    q_v>0. Isca B_FRIERSON has NO such term (q_v=None recovers it).
    """
    T, _pf, p_half, _ts, lat, q_v, _insol = _columns()
    p_s = p_half[:, -1]
    # Pin the SHIPPED default coeff (canary): the moisture departure is the
    # default behaviour, not merely a value we override here.
    assert GrayRadiationConfig().tau_moist_coeff == 0.0115
    cfg = _isca_matched_config()._replace(tau_moist_coeff=0.0115)

    dtau_dry = np.asarray(
        _compute_lw_optical_depth(
            jnp.asarray(p_half), jnp.asarray(p_s), jnp.asarray(lat), None, cfg,
        )
    )
    dtau_moist = np.asarray(
        _compute_lw_optical_depth(
            jnp.asarray(p_half), jnp.asarray(p_s), jnp.asarray(lat),
            jnp.asarray(q_v), cfg,
        )
    )
    dp = np.diff(p_half, axis=1)
    added = cfg.tau_moist_coeff * q_v * dp / constants.g
    assert np.allclose(dtau_moist - dtau_dry, added, rtol=1e-10, atol=1e-14)
    assert np.all(dtau_moist >= dtau_dry)
    assert np.any(dtau_moist > dtau_dry)  # q_v>0 somewhere
