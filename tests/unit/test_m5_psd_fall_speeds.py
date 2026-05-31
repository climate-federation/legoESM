"""SAM M2005 PSD mass-weighted fall-speed tests (iter-15 M5).

gSAM ``MICRO_M2005`` sediments each species at the mass-weighted moment
of its PSD (``module_mp_graupel.f90`` slopes 1640/2549, fall speeds
1854-1855 / 4230-4238):

    LAMR = (π·ρ_w·N_r/q_r)^⅓,   LAMI = (ρ_ci·π·N_i/q_i)^⅓
    UM   = a·Γ(4+b)/6 · LAM^−b · (ρ_su/ρ)^0.54   (with SAM caps)

The faithful ``fall_speed_scheme="m2005_psd"`` couples rain/ice fall
speed to the PROGNOSTIC number (N_r, N_i): at fixed mass, more drops ⇒
smaller drops ⇒ slower fall. The legacy ``"bulk_qpower"`` form
``V_t=a_v·(q·ρ/ρ_sfc)^b_v`` ignores number entirely. Cloud ice / rain
are double-moment in legoESM; snow has no N_s, so it stays bulk.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)

# Two-level warm column (no ice); rain sits in the BOTTOM cell so one step
# of sedimentation produces an immediately observable surface flux ∝ V_t.
_T = 290.0
_P_FULL = jnp.asarray([[7.0e4, 9.0e4]])           # (1, 2)
_P_HALF = jnp.asarray([[6.0e4, 8.0e4, 1.0e5]])    # (1, 3)
_RHO = _P_FULL / (constants.R_d * _T)
_DZ = jnp.full((1, 2), 300.0)                      # layer thickness [m]


def _precip(q_r=2.0e-3, N_r=1.0e4, scheme="m2005_psd", q_i=0.0, N_i=0.0,
            dt=20.0):
    """Surface precip flux from a 2-level column with rain in the base cell.

    q_v is set to liquid saturation so rain evaporation is inert and the
    only q_r process is sedimentation — precip is a clean fall-speed probe.
    """
    qsat = saturation_mixing_ratio(jnp.asarray(_T), _P_FULL)
    q_v = qsat                                     # RH=1 ⇒ no evap/cond
    z = jnp.zeros((1, 2))
    # Rain only in the bottom level (index -1).
    qr = z.at[0, -1].set(q_r)
    nr = z.at[0, -1].set(N_r)
    qi = z.at[0, -1].set(q_i)
    ni = z.at[0, -1].set(N_i)
    hm = HydrometeorState(
        q_c=z, q_r=qr, q_i=qi, q_s=z, q_g=z,
        N_c=z, N_r=nr, N_i=ni,
    )
    out = morrison_microphysics(
        jnp.full((1, 2), _T), q_v, hm, _P_FULL, _P_HALF,
        _RHO, _DZ, dt,
        MorrisonConfig(fall_speed_scheme=scheme, N_i0=0.0),
    )
    return float(out.precipitation[0])


def test_psd_rain_precip_positive():
    """Rain in the base cell sediments out ⇒ positive surface precip."""
    assert _precip(q_r=2.0e-3, N_r=1.0e4) > 0.0


def test_psd_more_drops_fall_slower():
    """Faithful signature: at FIXED rain mass, MORE drops (higher N_r) ⇒
    smaller drops ⇒ slower mass-weighted fall ⇒ LESS surface precip in one
    step. (LAMR ∝ N_r^⅓, UMR ∝ LAMR^−0.8 ∝ N_r^−0.27.)"""
    few = _precip(q_r=2.0e-3, N_r=1.0e3)     # few, big drops → fast
    many = _precip(q_r=2.0e-3, N_r=1.0e6)    # many, small drops → slow
    assert few > many


def test_psd_more_mass_falls_faster():
    """At FIXED number, more rain mass ⇒ bigger drops ⇒ faster fall."""
    light = _precip(q_r=5.0e-4, N_r=1.0e4)
    heavy = _precip(q_r=5.0e-3, N_r=1.0e4)
    # Heavier rain has both more mass AND faster drops, so flux must rise.
    assert heavy > light


def test_bulk_qpower_ignores_number():
    """The legacy bulk scheme's fall speed does NOT depend on N_r, so
    precip is invariant to drop number — the very deficiency m2005_psd
    fixes."""
    lo = _precip(q_r=2.0e-3, N_r=1.0e3, scheme="bulk_qpower")
    hi = _precip(q_r=2.0e-3, N_r=1.0e6, scheme="bulk_qpower")
    assert lo == pytest.approx(hi, rel=1e-12)


def test_psd_differs_from_bulk():
    """The two schemes give materially different sedimentation for the
    same state (this default change is physically meaningful)."""
    psd = _precip(q_r=2.0e-3, N_r=1.0e4, scheme="m2005_psd")
    bulk = _precip(q_r=2.0e-3, N_r=1.0e4, scheme="bulk_qpower")
    assert abs(psd - bulk) > 1.0e-6


def test_psd_empty_cell_no_precip_no_nan():
    """Zero rain ⇒ exactly zero precip, no NaN from the LAM divide."""
    p = _precip(q_r=0.0, N_r=0.0)
    assert p == 0.0


def test_psd_fall_speed_ad_safe():
    """jax.grad of surface precip wrt rain mass is finite at q_r=0 (the
    1e-20 slope floor + jnp.where mask must not leak inf/nan adjoints)."""
    qsat = saturation_mixing_ratio(jnp.asarray(_T), _P_FULL)

    def loss(qr_base):
        z = jnp.zeros((1, 2))
        qr = z.at[0, -1].set(qr_base)
        nr = z.at[0, -1].set(1.0e4)
        hm = HydrometeorState(
            q_c=z, q_r=qr, q_i=z, q_s=z, q_g=z,
            N_c=z, N_r=nr, N_i=z,
        )
        out = morrison_microphysics(
            jnp.full((1, 2), _T), qsat, hm, _P_FULL, _P_HALF,
            _RHO, _DZ, 20.0,
            MorrisonConfig(N_i0=0.0),
        )
        return jnp.sum(out.precipitation)

    for qr0 in (0.0, 1.0e-3):
        g = jax.grad(loss)(jnp.asarray(qr0))
        assert bool(jnp.isfinite(g))


def test_psd_ice_precip_couples_to_number():
    """Cloud ice (double-moment) sediments via LAMI(N_i): at fixed q_i,
    more crystals ⇒ smaller ⇒ slower ⇒ less ice precip."""
    # Cold column so the deposited/sedimenting species is ice, not rain.
    qsat = saturation_mixing_ratio(jnp.asarray(240.0), _P_FULL)

    def ice_precip(N_i):
        z = jnp.zeros((1, 2))
        qi = z.at[0, -1].set(1.0e-4)
        ni = z.at[0, -1].set(N_i)
        hm = HydrometeorState(
            q_c=z, q_r=z, q_i=qi, q_s=z, q_g=z,
            N_c=z, N_r=z, N_i=ni,
        )
        out = morrison_microphysics(
            jnp.full((1, 2), 240.0), 0.5 * qsat, hm, _P_FULL, _P_HALF,
            _RHO, _DZ, 20.0,
            MorrisonConfig(N_i0=0.0),
        )
        return float(out.precipitation[0])

    few = ice_precip(1.0e3)
    many = ice_precip(1.0e7)
    assert few > many >= 0.0


def test_psd_column_water_conserved_with_extra_sink():
    """Codex iter-15 D: with the faster PSD fall speeds AND a competing
    in-column sink (melting near 0 °C), column-integrated water still
    closes: Σ_k ρ_k·(Σ dq)_k·dz_k = −precip (the only water leaving is the
    surface flux). Stresses the joint sedimentation+extra_sink donor clamp
    at a large dt with rain, ice and snow all present."""
    T = 274.0                                  # melting active ⇒ extra_sink≠0
    qsat = saturation_mixing_ratio(jnp.asarray(T), _P_FULL)
    z = jnp.zeros((1, 2))
    qr = z.at[0, -1].set(3.0e-3)
    qi = z.at[0, -1].set(5.0e-4)
    qs = z.at[0, -1].set(1.0e-3)
    nr = z.at[0, -1].set(1.0e4)
    ni = z.at[0, -1].set(1.0e5)
    hm = HydrometeorState(
        q_c=z, q_r=qr, q_i=qi, q_s=qs, q_g=z,
        N_c=z, N_r=nr, N_i=ni,
    )
    out = morrison_microphysics(
        jnp.full((1, 2), T), qsat, hm, _P_FULL, _P_HALF,
        _RHO, _DZ, 60.0, MorrisonConfig(N_i0=0.0),
    )
    dq_tot = (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt
              + out.dq_i_dt + out.dq_s_dt + out.dq_g_dt)
    column_tend = float(jnp.sum(_RHO * _DZ * dq_tot))   # kg/m²/s
    residual = column_tend + float(out.precipitation[0])
    # Normalise by the precip magnitude; closure to a tight tolerance.
    assert abs(residual) < 1.0e-6 * (abs(float(out.precipitation[0])) + 1.0e-12)


def test_unknown_fall_speed_scheme_raises():
    with pytest.raises(ValueError, match="Unknown fall_speed_scheme"):
        _precip(scheme="stokes_typo")


def test_ice_fall_exponent_matches_gsam():
    """gSAM (the oracle) uses the MK-tuned cloud-ice fall exponent
    ``clice_fall_b = 0.865`` (``micro_params.f90:62``; ``BI`` in
    ``module_mp_graupel.f90:430``), NOT the M2005-ORIGINAL 1.0. The original
    1.0 fell ~3x too slow (v=0.01/0.07/0.7 vs gSAM 0.03/0.24/1.78 m/s) ⇒ anvil
    cloud-ice over-accumulated (iter-205). Guard the faithful value + the speed
    direction (rain/snow/graupel exponents already matched gSAM)."""
    import math
    cfg = MorrisonConfig()
    assert cfg.fall_b_i == pytest.approx(0.865), (
        f"cloud-ice fall exponent must match gSAM clice_fall_b=0.865, "
        f"got {cfg.fall_b_i}"
    )
    # Mass-weighted V_t_i = a·Γ(4+b)/6·lami^(−b). For a real ice PSD slope
    # (lami ~ 1e4–1e5 /m) the gSAM b=0.865 falls markedly FASTER than the
    # M2005-original b=1.0 (the over-accumulation fix).
    lami = 3.0e4
    v_gsam = math.gamma(4.0 + cfg.fall_b_i) / 6.0 * lami ** (-cfg.fall_b_i)
    v_m2005_orig = math.gamma(5.0) / 6.0 * lami ** (-1.0)
    assert v_gsam > 2.0 * v_m2005_orig
