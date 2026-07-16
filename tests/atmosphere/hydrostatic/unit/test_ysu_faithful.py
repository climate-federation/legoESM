"""Scheme-level oracle-faithfulness tests for the YSU (Hong et al. 2006) PBL scheme.

Pins the faithful Hong06 / Troen-Mahrt structure and locks the differentiable
flux-matched-entrainment surrogate as canaries.

FAITHFUL (Hong06 / Troen-Mahrt forms + constants):
  * the K-profile shape (z/h)(1−z/h)² peaks at exactly 4/27 at z=h/3
    (_KPROFILE_PEAK_FRAC, the entrainment-K stability-cap constant);
  * the bulk-Ri PBL height with a Troen-Mahrt thermal-excess parcel deepens h_pbl
    under unstable surface forcing;
  * the entrainment COEFFICIENT e_ratio=0.15 (the Hong06 free-convective value).

DEPARTURE / SURROGATE (locked + labeled):
  * the IMPLEMENTED fixed surface-flux-ratio entrainment (w'θ')_h = −e_ratio·(w'θ')_0
    is only Hong06's FREE-CONVECTIVE limit (the published entrainment velocity uses
    w_m³=w*³+5u*³, shear-dependent). ΔKh = Kh(e_ratio)−Kh(0) isolates the entrainment
    diffusivity exactly (only entrainment depends on e_ratio) and pins that it is
    LINEAR in e_ratio below the cap — NOT the full published closure;
  * the flux-matched entrainment K carries a stability cap (4/27)·κ·w_s(h)·h — a
    huge e_ratio SATURATES the entrainment K (breaks the linearity), a cap canary.

Auxiliary sanity + hardening (not faithfulness pins): Km, Kh ≥ 0; an unknown
turbulence scheme raises ValueError (dispatch hardening).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig, YSUConfig
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn
from legoesm.atmosphere.physics.turbulence.ysu import (
    _KPROFILE_PEAK_FRAC,
    ysu_turbulence,
)
from legoesm.thermo import saturation_mixing_ratio

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _convective_column(ncol=2, nlev=30):
    """A convective BL: well-mixed θ below a sharp inversion, warm free troposphere.

    Level 0 is the top, nlev-1 the surface. Returns the full ysu_turbulence input
    tuple plus z_full and h-scale for locating the PBL.
    """
    p_half = jnp.broadcast_to(
        jnp.linspace(2.0e4, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1)
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    frac = jnp.linspace(0.0, 1.0, nlev)          # 0 at top, 1 at surface
    # θ ~ 300 K well-mixed below (frac>0.6); warmer aloft (stable free troposphere)
    theta = jnp.where(frac > 0.6, 300.0, 300.0 + (0.6 - frac) * 60.0)
    theta = jnp.broadcast_to(theta[None, :], (ncol, nlev))
    T = theta * (p_full / constants.p_ref) ** constants.kappa
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None)))
    z_half = jnp.concatenate(
        [jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((ncol, 1))], axis=1
    )
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    rho = p_full / (constants.R_d * T)
    u = jnp.broadcast_to(jnp.linspace(10.0, 3.0, nlev)[None, :], (ncol, nlev))
    v = jnp.full((ncol, nlev), 1.0)
    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
    return u, v, T, q_v, p_full, p_half, z_full, z_half, rho


def _run(e_ratio, T_sfc_K=305.0, config_kw=None):
    u, v, T, q_v, p_full, p_half, z_full, z_half, rho = _convective_column()
    T_sfc = jnp.full((u.shape[0],), T_sfc_K)
    q_sfc = saturation_mixing_ratio(T_sfc, p_half[:, -1])
    kw = {"entrainment_ratio": e_ratio}
    if config_kw:
        kw.update(config_kw)
    cfg = YSUConfig(**kw)
    out = ysu_turbulence(
        u, v, T, q_v, p_full, p_half, z_full, z_half, T_sfc, q_sfc, rho, 300.0, cfg,
    )
    return out, np.asarray(z_full)[0]


def _dKh(e_ratio, T_sfc_K=305.0):
    """Entrainment-only Kh contribution: Kh(e_ratio) − Kh(0) (col 0)."""
    o_e, z = _run(e_ratio, T_sfc_K)
    o_0, _ = _run(0.0, T_sfc_K)
    return np.asarray(o_e.Kh)[0] - np.asarray(o_0.Kh)[0], z, float(o_e.h_pbl[0])


# ===========================================================================
# FAITHFUL forms
# ===========================================================================
def test_faithful_ysu_entrainment_coefficient_is_hong06_value():
    """Canary: e_ratio = 0.15, the Hong06 free-convective entrainment coefficient."""
    assert YSUConfig().entrainment_ratio == 0.15


def test_ysu_ri_crit_is_local_default():
    """Ri_crit = 0.25 is a LOCAL config default, not a universal Hong06 constant.

    WRF YSU uses a regime-dependent critical bulk Richardson number (unstable
    brcr_ub=0.0, plus stable/land branches near 0.25); this scheme carries a single
    0.25 default. Canary for the config default, not an oracle constant.
    """
    assert YSUConfig().Ri_crit == 0.25


def test_ysu_entrainment_cap_fraction_provenance():
    """_KPROFILE_PEAK_FRAC (the entrainment-K cap fraction) is the Troen-Mahrt peak.

    The entrainment stability cap is (4/27)·κ·w_s(h)·h, where 4/27 is the peak of the
    K-profile shape (z/h)(1−z/h)² at z/h=1/3. This pins the CAP constant's provenance
    (a wrong value like a stray 0.15 would fail) — NOT the K-profile shape itself,
    which enters Km_profile directly rather than through this constant.
    """
    x = np.linspace(0.0, 1.0, 200001)
    shape = x * (1.0 - x) ** 2
    assert x[shape.argmax()] == pytest.approx(1.0 / 3.0, abs=1e-4)   # peak at z/h=1/3
    assert shape.max() == pytest.approx(4.0 / 27.0, rel=1e-8)
    assert _KPROFILE_PEAK_FRAC == pytest.approx(4.0 / 27.0, rel=1e-12)


def test_faithful_ysu_pbl_deepens_with_unstable_forcing():
    """Troen-Mahrt thermal excess: a warmer surface (stronger w'θ'_0) deepens h_pbl."""
    o_cool, _ = _run(0.15, T_sfc_K=299.0)
    o_warm, _ = _run(0.15, T_sfc_K=307.0)
    assert float(o_warm.h_pbl[0]) > float(o_cool.h_pbl[0])
    assert float(o_warm.shflx[0]) > float(o_cool.shflx[0]) > 0.0  # unstable, upward


# ===========================================================================
# DEPARTURES (differentiable surrogate / numerics)
# ===========================================================================
def test_ysu_entrainment_fixed_ratio_surrogate_is_linear():
    """The IMPLEMENTED fixed-ratio entrainment surrogate is LINEAR in e_ratio.

    ΔKh = Kh(e_ratio) − Kh(0) isolates the entrainment diffusivity exactly (only the
    entrainment term depends on e_ratio). Doubling e_ratio doubles ΔKh at its peak,
    the signature of the fixed-fraction-of-surface-flux law (w'θ')_h=−e_ratio·(w'θ')_0.
    This pins the implemented surrogate's LINEARITY — NOT the full Hong06 closure,
    whose entrainment velocity w_m³=w*³+5u*³ makes the true top flux shear-dependent.
    """
    dkh1, _, _ = _dKh(0.15)
    dkh2, _, _ = _dKh(0.30)
    pk = dkh1.argmax()
    assert dkh1[pk] > 0.0                                   # entrainment adds mixing
    assert dkh2[pk] / dkh1[pk] == pytest.approx(2.0, rel=1e-9)


def test_ysu_entrainment_isolated_and_upper_bl():
    """Surrogate canary: the flux-matched entrainment K is isolated by e_ratio and
    concentrated in the upper BL (an implementation property of the K realization,
    not a Hong06 faithfulness pin)."""
    dkh, z, h_pbl = _dKh(0.15)
    assert np.all(dkh >= -1e-12)                            # entrainment never removes K
    # peak sits in the upper part of the boundary layer (near the inversion)
    z_peak = z[dkh.argmax()]
    assert z_peak > 0.5 * h_pbl
    assert z_peak <= h_pbl * 1.2


def test_departure_ysu_entrainment_stability_cap():
    """Stability cap: a huge e_ratio SATURATES the entrainment K (breaks linearity).

    The flux-matched K may not exceed (4/27)·κ·w_s(h)·h; once capped, doubling
    e_ratio no longer doubles ΔKh (the surrogate departure from the pure ratio
    closure), so the 2× ratio collapses toward 1.0.
    """
    dkh_a, _, _ = _dKh(2.0)
    dkh_b, _, _ = _dKh(4.0)
    pk = dkh_a.argmax()
    assert dkh_a[pk] > 0.0
    assert dkh_b[pk] / dkh_a[pk] == pytest.approx(1.0, abs=1e-6)   # fully capped
    assert dkh_b[pk] / dkh_a[pk] < 1.5                            # NOT the 2× ratio


def test_ysu_diffusivities_nonnegative():
    """Km, Kh ≥ 0 everywhere (positivity of the eddy diffusivities)."""
    o, _ = _run(0.15)
    assert np.all(np.asarray(o.Km) >= 0.0)
    assert np.all(np.asarray(o.Kh) >= 0.0)
    assert np.all(np.isfinite(np.asarray(o.Kh)))


def test_dispatch_unknown_turbulence_raises():
    """Dispatch hardening: an unknown turbulence scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown turbulence scheme"):
        get_turbulence_fn(TurbulenceConfig(scheme="not_a_scheme"))
